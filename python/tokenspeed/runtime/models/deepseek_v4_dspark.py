# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Inference-only DeepSeek V4 DSpark draft model."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable

import torch
from tokenspeed_kernel import mhc_fused_hc, mhc_post, mhc_pre
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.utils import get_moe_backend
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4Compressor,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4MegaMoEExperts,
    _deepseek_v4_expert_scale_parameter_name,
    hc_head,
)
from tokenspeed.runtime.models.deepseek_v4_dspark_ops.attention import (
    _dspark_fp8_linear,
    _quantize_dspark_non_rope,
    _rmsnorm,
    _rope_last_dims_batched,
    dspark_attention_forward_batched,
    precompute_dspark_freqs_cis,
)
from tokenspeed.runtime.models.deepseek_v4_dspark_ops.heads import (
    DSparkConfidenceHead,
    DSparkVanillaMarkov,
)
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)

_DSPARK_WEIGHT_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")
DEFAULT_DSPARK_WINDOW_SIZE = 128
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")
_ATTENTION_TENSORS = frozenset(
    {
        "attn_sink",
        "kv_norm.weight",
        "q_norm.weight",
        "wkv.scale",
        "wkv.weight",
        "wo_a.scale",
        "wo_a.weight",
        "wo_b.scale",
        "wo_b.weight",
        "wq_a.scale",
        "wq_a.weight",
        "wq_b.scale",
        "wq_b.weight",
    }
)
_ATTENTION_CHECKPOINT_TENSORS = frozenset(f"attn.{name}" for name in _ATTENTION_TENSORS)
_STAGE_ZERO_CORE = frozenset(
    {"main_norm.weight", "main_proj.scale", "main_proj.weight"}
)
_STAGE_COMMON_CORE = frozenset(
    {
        "attn_norm.weight",
        "ffn.gate.bias",
        "ffn.gate.weight",
        "ffn.shared_experts.w1.scale",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w2.scale",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w3.scale",
        "ffn.shared_experts.w3.weight",
        "ffn_norm.weight",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    }
)
_LAST_STAGE_CORE = frozenset(
    {
        "confidence_head.proj.weight",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "norm.weight",
    }
)
_ZERO_INITIALIZED_EXPERT_BIAS_SUFFIXES = (
    ".experts.w13_weight_bias",
    ".experts.w2_weight_bias",
)
_REPLICATE_VOCAB_HEADS_ENV = "TOKENSPEED_DSPARK_REPLICATE_VOCAB_HEADS"
_REPLICATE_MARKOV_EMBEDDING_ENV = "TOKENSPEED_DSPARK_REPLICATE_MARKOV_EMBEDDING"


def _is_zero_initialized_expert_bias(name: str) -> bool:
    return name.endswith(_ZERO_INITIALIZED_EXPERT_BIAS_SUFFIXES)


def _replicate_dspark_vocab_weight(
    local_weight: torch.Tensor,
    module: VocabParallelEmbedding,
    mapping: Mapping,
    *,
    label: str,
) -> torch.Tensor:
    """Reconstruct an unpadded vocabulary weight in TP-rank order.

    This candidate intentionally supports only the DeepSeek V4 checkpoint
    layout with no base-vocabulary padding or added vocabulary. Other layouts
    need an explicit reindexing step and therefore fail closed here.
    """

    tp_size = int(mapping.attn.tp_size)
    if int(module.tp_size) != 1:
        raise ValueError(f"Replicated DSpark {label} module must use tp_size=1.")
    if local_weight.ndim != 2:
        raise ValueError(
            f"DSpark target {label} weight must be rank 2; "
            f"got {tuple(local_weight.shape)}."
        )
    if int(module.num_added_embeddings) != 0 or int(
        module.org_vocab_size_padded
    ) != int(module.org_vocab_size):
        raise ValueError(
            f"Replicated DSpark {label} requires an unpadded base-only vocabulary."
        )
    expected_full_rows = int(module.num_embeddings_padded)
    if expected_full_rows % tp_size != 0:
        raise ValueError(
            f"Replicated DSpark {label} vocabulary size {expected_full_rows} "
            f"is not divisible by TP size {tp_size}."
        )
    expected_local_rows = expected_full_rows // tp_size
    if local_weight.shape[0] != expected_local_rows:
        raise ValueError(
            f"DSpark target {label} shard has {local_weight.shape[0]} rows; "
            f"expected {expected_local_rows} for TP{tp_size}."
        )
    full_weight = all_gather(
        local_weight.contiguous(),
        mapping.attn.tp_group,
        dim=0,
    )
    expected_shape = (expected_full_rows, local_weight.shape[1])
    if tuple(full_weight.shape) != expected_shape:
        raise RuntimeError(
            f"DSpark replicated {label} shape mismatch: "
            f"expected {expected_shape}, got {tuple(full_weight.shape)}."
        )
    return full_weight.contiguous()


def count_dspark_stages(
    model_path: str,
    revision: str | None = None,
) -> int | None:
    """Count contiguous ``mtp.<stage>`` namespaces in a safetensors index."""

    index_filename = "model.safetensors.index.json"
    if os.path.isdir(model_path):
        index_path = os.path.join(model_path, index_filename)
    else:
        from huggingface_hub import hf_hub_download

        try:
            index_path = hf_hub_download(
                repo_id=model_path,
                filename=index_filename,
                revision=revision,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed below
            logger.debug(
                "Unable to resolve DSpark safetensors index for %s: %s",
                model_path,
                exc,
            )
            return None
    if not os.path.isfile(index_path):
        return None
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map", {})
    stages = {
        int(match.group(1))
        for name in weight_map
        if (match := _DSPARK_WEIGHT_RE.match(name))
    }
    if not stages:
        return None
    expected = set(range(max(stages) + 1))
    if stages != expected:
        raise ValueError(
            "DSpark checkpoint stages must be contiguous from zero; "
            f"found {sorted(stages)}."
        )
    return len(stages)


def _block_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a DeepSeek block-scaled FP8 matrix to BF16."""

    rows, columns = weight.shape
    expanded_scale = scale.float().repeat_interleave(block_size, 0)[:rows]
    expanded_scale = expanded_scale.repeat_interleave(block_size, 1)[:, :columns]
    return (weight.float() * expanded_scale).to(torch.bfloat16)


def _apply_dspark_hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Apply the token-wise mHC head while preserving the DSpark block axes."""

    if hidden_states.ndim != 4:
        raise ValueError(
            "DSpark mHC head expects [batch, block, streams, hidden], "
            f"got {tuple(hidden_states.shape)}."
        )
    batch_size, block_size, hc_mult, hidden_size = hidden_states.shape
    output = hc_head(
        hidden_states.reshape(-1, hc_mult, hidden_size),
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps,
        hc_eps,
    )
    return output.reshape(batch_size, block_size, hidden_size)


class _DSparkStage(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        *,
        stage_id: int,
        num_stages: int,
        num_capture_layers: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.block = DeepseekV4DecoderLayer(
            config,
            int(config.num_hidden_layers) + stage_id,
            mapping,
            quant_config,
            add_prefix("block", prefix),
            cache_layer_index=stage_id,
        )
        if stage_id == 0:
            self.main_proj = ReplicatedLinear(
                int(config.hidden_size) * num_capture_layers,
                int(config.hidden_size),
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("main_proj", prefix),
            )
            self.main_norm = RMSNorm(
                int(config.hidden_size),
                eps=float(config.rms_norm_eps),
            )
        if stage_id == num_stages - 1:
            self.norm = RMSNorm(
                int(config.hidden_size),
                eps=float(config.rms_norm_eps),
            )
            hc_mult = int(config.hc_mult)
            hidden_size = int(config.hidden_size)
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_mult * hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(hc_mult, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=False,
            )


class DeepseekV4DSparkModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.num_stages = int(getattr(config, "dspark_num_stages", 0))
        if self.num_stages <= 0:
            raise ValueError(
                "DSpark requires a positive checkpoint-derived stage count."
            )
        self.block_size = int(getattr(config, "dspark_block_size", 0))
        if self.block_size <= 0:
            raise ValueError("DSpark checkpoint must define dspark_block_size.")
        self.target_layer_ids = tuple(
            int(layer_id) for layer_id in getattr(config, "dspark_target_layer_ids", ())
        )
        if not self.target_layer_ids:
            raise ValueError("DSpark checkpoint must define dspark_target_layer_ids.")
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("DSpark target layer IDs must be unique.")
        if tuple(sorted(self.target_layer_ids)) != self.target_layer_ids:
            raise ValueError("DSpark target layer IDs must be strictly increasing.")
        if self.target_layer_ids[-1] >= int(config.num_hidden_layers):
            raise ValueError(
                "DSpark target layer IDs must refer to target decoder layers."
            )

        self.hidden_size = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.hc_eps = float(config.hc_eps)
        self.noise_token_id = int(getattr(config, "dspark_noise_token_id"))
        self.markov_rank = int(getattr(config, "dspark_markov_rank", 0))
        if self.markov_rank <= 0:
            raise ValueError("Week-0 DSpark requires a positive vanilla Markov rank.")
        markov_kind = getattr(config, "dspark_markov_head_type", None) or "vanilla"
        if str(markov_kind).lower() != "vanilla":
            raise ValueError(
                "Week-0 DSpark supports only the vanilla Markov head; "
                f"got {markov_kind!r}."
            )
        self.replicate_vocab_heads = os.getenv(_REPLICATE_VOCAB_HEADS_ENV, "0") == "1"
        self.replicate_markov_embedding = self.replicate_vocab_heads or (
            os.getenv(_REPLICATE_MARKOV_EMBEDDING_ENV, "0") == "1"
        )
        target_vocab_parallel_kwargs = (
            {}
            if self.replicate_vocab_heads
            else {
                "tp_rank": mapping.attn.tp_rank,
                "tp_size": mapping.attn.tp_size,
                "tp_group": mapping.attn.tp_group,
            }
        )
        markov_embedding_parallel_kwargs = (
            {} if self.replicate_markov_embedding else target_vocab_parallel_kwargs
        )

        self.stages = nn.ModuleList(
            [
                _DSparkStage(
                    config,
                    mapping,
                    quant_config,
                    stage_id=stage_id,
                    num_stages=self.num_stages,
                    num_capture_layers=len(self.target_layer_ids),
                    prefix=add_prefix(f"stages.{stage_id}", prefix),
                )
                for stage_id in range(self.num_stages)
            ]
        )
        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size),
            self.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
            **target_vocab_parallel_kwargs,
        )
        self.markov_embedding = VocabParallelEmbedding(
            int(config.vocab_size),
            self.markov_rank,
            params_dtype=torch.float32,
            prefix=add_prefix("markov_embedding", prefix),
            **markov_embedding_parallel_kwargs,
        )
        self.markov_projection = ParallelLMHead(
            int(config.vocab_size),
            self.markov_rank,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("markov_projection", prefix),
            **target_vocab_parallel_kwargs,
        )
        self.markov_head = DSparkVanillaMarkov(
            self.markov_embedding,
            self.markov_projection,
        )
        logger.info(
            "DSPARK_VOCAB_LAYOUT rank=%d target_embedding_tp=%d "
            "target_lm_head_tp=pending markov_embedding_tp=%d "
            "markov_projection_tp=%d",
            mapping.rank,
            self.embed_tokens.tp_size,
            self.markov_embedding.tp_size,
            self.markov_projection.tp_size,
        )
        self.confidence_projection = ReplicatedLinear(
            self.hidden_size + self.markov_rank,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("confidence_projection", prefix),
        )
        self.confidence_head = DSparkConfidenceHead(self.confidence_projection)
        self.register_buffer("_local_base_head_fp32", None, persistent=False)
        self._local_base_head_source_ptr: int | None = None
        self._local_base_head_source_version: int | None = None

        head_dim = int(getattr(config, "head_dim"))
        self.attention_params = {
            "n_heads": int(config.num_attention_heads),
            "head_dim": head_dim,
            "rope_head_dim": int(config.qk_rope_head_dim),
            "n_groups": int(config.o_groups),
            "o_lora_rank": int(config.o_lora_rank),
            "window_size": int(
                getattr(config, "dspark_window_size", DEFAULT_DSPARK_WINDOW_SIZE)
            ),
            "eps": self.rms_norm_eps,
            "softmax_scale": head_dim**-0.5,
        }
        frequency_capacity = (
            int(getattr(config, "max_position_embeddings")) + self.block_size + 2
        )
        frequency_device = self.stages[0].block.attn_norm.weight.device
        self.register_buffer(
            "freqs_cis",
            precompute_dspark_freqs_cis(
                self.attention_params["rope_head_dim"],
                frequency_capacity,
                float(getattr(config, "rope_theta", 10000.0)),
                frequency_device,
            ),
            persistent=False,
        )

    @property
    def window_size(self) -> int:
        return int(self.attention_params["window_size"])

    def _forward_stage(
        self,
        stage: _DSparkStage,
        hidden_states: torch.Tensor,
        main_x: torch.Tensor,
        start_pos: torch.Tensor,
        kv_windows: torch.Tensor,
        slots: torch.Tensor,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        batch, block_size, _, hidden_size = hidden_states.shape
        block = stage.block
        residual = hidden_states
        layer_input, post, comb = mhc_pre(
            residual,
            block.hc_attn_fn,
            block.hc_attn_scale,
            block.hc_attn_base,
            block.rms_norm_eps,
            block.hc_eps,
            block.hc_sinkhorn_iters,
            norm_weight=None,
            norm_eps=None,
        )
        layer_input = block.attn_norm(layer_input)
        attention_output = dspark_attention_forward_batched(
            layer_input,
            main_x,
            start_pos,
            kv_windows[:, stage.stage_id],
            slots,
            freqs_cis=self.freqs_cis,
            **stage._dspark_attn,
            **self.attention_params,
        )
        residual, layer_input, post, comb = mhc_fused_hc(
            attention_output,
            residual,
            post,
            comb,
            block.hc_ffn_fn,
            block.hc_ffn_scale,
            block.hc_ffn_base,
            block.rms_norm_eps,
            block.hc_eps,
            block.hc_sinkhorn_iters,
            norm_weight=None,
            norm_eps=None,
        )
        layer_input = block.ffn_norm(layer_input)
        flat_input = layer_input.reshape(batch * block_size, hidden_size)
        flat_ids = input_ids.reshape(-1)
        use_mega_moe = getattr(block.ffn, "use_mega_moe", False)
        if use_mega_moe:
            ffn_output = block.ffn(
                flat_input,
                flat_ids,
                batch * block_size,
                batch * block_size,
                ctx=ctx,
                comm_manager=block.comm_manager,
            )
        else:
            flat_input = block.comm_manager.pre_mlp_comm(flat_input, ctx)
            if block.ffn.gate.is_hash_moe:
                flat_ids = block._pre_mlp_input_ids_comm(flat_ids, ctx)
            num_global_tokens, max_num_tokens_per_gpu = (
                block.comm_manager.get_num_tokens(ctx)
            )
            ffn_output = block.ffn(
                flat_input,
                flat_ids,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
            ffn_output, _ = block.comm_manager.post_mlp_comm(ffn_output, None, ctx)
        return mhc_post(
            ffn_output.reshape(batch, block_size, hidden_size),
            residual,
            post,
            comb,
        )

    def forward_backbone(
        self,
        captured_hidden_states: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        start_pos: torch.Tensor,
        kv_windows: torch.Tensor,
        slots: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        expected_width = len(self.target_layer_ids) * self.hidden_size
        if (
            captured_hidden_states.ndim != 2
            or captured_hidden_states.shape[1] != expected_width
        ):
            raise ValueError(
                "DSpark captured hidden-state width mismatch: "
                f"expected {expected_width}, got {tuple(captured_hidden_states.shape)}."
            )
        stage_zero = self.stages[0]
        main_x, _ = stage_zero.main_proj(captured_hidden_states)
        main_x = stage_zero.main_norm(main_x).unsqueeze(1)
        draft_ids = bonus_token_ids.new_full(
            (bonus_token_ids.shape[0], self.block_size),
            self.noise_token_id,
        )
        draft_ids[:, 0] = bonus_token_ids
        hidden_states = self.embed_tokens(draft_ids)
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, 1, self.hc_mult, 1)
        for stage in self.stages:
            hidden_states = self._forward_stage(
                stage,
                hidden_states,
                main_x,
                start_pos,
                kv_windows,
                slots,
                draft_ids,
                ctx,
            )
        last_stage = self.stages[-1]
        hidden_states = _apply_dspark_hc_head(
            hidden_states,
            last_stage.hc_head_fn,
            last_stage.hc_head_scale,
            last_stage.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return last_stage.norm(hidden_states)

    def local_base_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module | None,
    ) -> torch.Tensor:
        """Compute public FP32 base logits from the local vocabulary shard.

        Production DSpark replay uses a stable FP32 head buffer initialized by
        ``set_embed_and_head``. Passing a head keeps the uncached reference
        path available; production callers pass ``None`` for the replay buffer.
        """

        if lm_head is not None:
            head_fp32 = lm_head.weight.float()
        else:
            head_fp32 = getattr(self, "_local_base_head_fp32", None)
            if head_fp32 is None:
                raise RuntimeError(
                    "DSpark local base logits require a cached target LM head."
                )
        return torch.matmul(hidden_states.float(), head_fp32.T)

    def refresh_local_base_logits_head(
        self,
        head: torch.Tensor,
        *,
        force: bool,
    ) -> bool:
        """Refresh the stable FP32 local-head buffer after a weight update.

        Returns ``True`` when the buffer was initialized or updated. Once the
        buffer exists, its storage address and shape stay fixed so CUDA Graph
        replays remain valid.
        """

        if head.ndim != 2:
            raise ValueError(
                "DSpark target LM-head weight must be rank 2; "
                f"got shape {tuple(head.shape)}."
            )
        source_ptr = head.data_ptr()
        source_version = int(head._version)
        cached = self._local_base_head_fp32
        if cached is None:
            cached = torch.empty(
                head.shape,
                dtype=torch.float32,
                device=head.device,
            )
            with torch.no_grad():
                cached.copy_(head)
            self._local_base_head_fp32 = cached
        elif not force and (
            source_ptr == self._local_base_head_source_ptr
            and source_version == self._local_base_head_source_version
        ):
            return False
        else:
            if cached.shape != head.shape or cached.device != head.device:
                raise RuntimeError(
                    "DSpark target LM-head shape or device changed after the "
                    "FP32 replay buffer was initialized."
                )
            with torch.no_grad():
                cached.copy_(head)
        self._local_base_head_source_ptr = source_ptr
        self._local_base_head_source_version = source_version
        return True

    def write_context_windows_batched(
        self,
        captured_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        valid: torch.Tensor,
        kv_windows: torch.Tensor,
        dummy_slot: int,
    ) -> None:
        del dummy_slot
        if captured_hidden_states.numel() == 0:
            return
        stage_zero = self.stages[0]
        main_x, _ = stage_zero.main_proj(captured_hidden_states.contiguous())
        main_x = stage_zero.main_norm(main_x)
        positions = positions.long()
        slots = slots.long()
        frequencies = self.freqs_cis[positions]
        rows = slots.unsqueeze(1).expand_as(positions)
        columns = positions % self.window_size
        valid_values = valid.unsqueeze(-1)
        for stage in self.stages:
            attention = stage._dspark_attn
            projected = _rmsnorm(
                _dspark_fp8_linear(main_x, attention["wkv"]),
                attention["kv_norm_w"],
                self.rms_norm_eps,
            )
            projected = _rope_last_dims_batched(
                projected,
                self.attention_params["rope_head_dim"],
                frequencies,
            )
            projected = _quantize_dspark_non_rope(
                projected,
                self.attention_params["rope_head_dim"],
            )
            stage_windows = kv_windows[:, stage.stage_id]
            current = stage_windows[rows, columns]
            stage_windows[rows, columns] = torch.where(
                valid_values,
                projected.to(stage_windows.dtype),
                current,
            )


class DeepseekV4ForCausalLMDSpark(nn.Module):
    """Draft-only DSpark model loaded from the target checkpoint."""

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.model = DeepseekV4DSparkModel(
            config,
            mapping,
            quant_config,
            add_prefix("model", prefix),
        )
        self.lm_head = ParallelLMHead(
            int(config.vocab_size),
            int(config.hidden_size),
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            **(
                {}
                if self.model.replicate_vocab_heads
                else {
                    "tp_rank": mapping.attn.tp_rank,
                    "tp_size": mapping.attn.tp_size,
                    "tp_group": mapping.attn.tp_group,
                }
            ),
        )
        logger.info(
            "DSPARK_TARGET_LM_HEAD_LAYOUT rank=%d target_lm_head_tp=%d",
            mapping.rank,
            self.lm_head.tp_size,
        )

    def get_hot_token_id(self):
        return None

    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        if self.model.replicate_vocab_heads:
            embed = _replicate_dspark_vocab_weight(
                embed,
                self.model.embed_tokens,
                self.mapping,
                label="embedding",
            )
            head = _replicate_dspark_vocab_weight(
                head,
                self.lm_head,
                self.mapping,
                label="LM head",
            )
            logger.info(
                "DSPARK_REPLICATED_VOCAB_HEADS_PASS rank=%d tp=%d "
                "embedding_shape=%s head_shape=%s",
                self.mapping.rank,
                self.mapping.attn.tp_size,
                tuple(embed.shape),
                tuple(head.shape),
            )
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        self.model.refresh_local_base_logits_head(head, force=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def refresh_replicated_embed_and_head(
        self,
        embed: torch.Tensor,
        head: torch.Tensor,
    ) -> None:
        """Refresh graph-stable replicated vocabulary buffers from TP shards."""

        if not self.model.replicate_vocab_heads:
            raise RuntimeError("DSpark vocabulary heads are not replicated.")
        replicated_embed = _replicate_dspark_vocab_weight(
            embed,
            self.model.embed_tokens,
            self.mapping,
            label="embedding",
        )
        replicated_head = _replicate_dspark_vocab_weight(
            head,
            self.lm_head,
            self.mapping,
            label="LM head",
        )
        destinations = (
            ("embedding", self.model.embed_tokens.weight, replicated_embed),
            ("LM head", self.lm_head.weight, replicated_head),
        )
        with torch.no_grad():
            for label, destination, source in destinations:
                if destination.shape != source.shape:
                    raise RuntimeError(
                        f"DSpark replicated {label} shape changed after initialization: "
                        f"expected {tuple(destination.shape)}, got {tuple(source.shape)}."
                    )
                destination.copy_(source)
        self.model.refresh_local_base_logits_head(self.lm_head.weight, force=True)

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        match = _DSPARK_WEIGHT_RE.match(name)
        return match is not None and int(match.group(1)) < self.model.num_stages

    @staticmethod
    def _map_stage_name(stage_id: int, suffix: str) -> str:
        prefix = f"model.stages.{stage_id}."
        if suffix == "main_norm.weight":
            return prefix + "main_norm.weight"
        if suffix.startswith("main_proj."):
            return prefix + suffix
        if suffix == "norm.weight":
            return prefix + "norm.weight"
        if suffix.startswith("hc_head_"):
            return prefix + suffix
        if suffix == "markov_head.markov_w1.weight":
            return "model.markov_embedding.weight"
        if suffix == "markov_head.markov_w2.weight":
            return "model.markov_projection.weight"
        if suffix == "confidence_head.proj.weight":
            return "model.confidence_projection.weight"
        if suffix == "attn_norm.weight":
            return prefix + "block.attn_norm.weight"
        if suffix == "ffn_norm.weight":
            return prefix + "block.ffn_norm.weight"
        if suffix.startswith(("hc_attn_", "hc_ffn_")):
            return prefix + "block." + suffix
        if suffix.startswith("attn."):
            return prefix + "block." + suffix
        if suffix.startswith("ffn."):
            return prefix + "block." + suffix
        return prefix + suffix

    def _map_checkpoint_name(self, raw_name: str) -> str | None:
        match = _DSPARK_WEIGHT_RE.match(raw_name)
        if match is None:
            return None
        stage_id = int(match.group(1))
        if stage_id >= self.model.num_stages:
            return None
        name = self._map_stage_name(stage_id, match.group(2))
        if name.endswith(".scale"):
            scale_suffix = ".weight_scale_inv"
            if _EXPERT_SCALE_RE.search(name):
                scale_suffix = "." + _deepseek_v4_expert_scale_parameter_name(
                    self.config,
                    use_mega_moe=get_moe_backend().is_mega_moe(),
                )
            name = name.removesuffix(".scale") + scale_suffix
        if ".shared_experts.w2" in name:
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if ".ffn.gate.bias" in name:
            name = name.replace(
                ".ffn.gate.bias",
                ".ffn.gate.e_score_correction_bias",
            )
        return name

    def get_stacked_params_mapping(self):
        return [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        stacked_mapping = self.get_stacked_params_mapping()
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1",
                down_proj_name="w2",
                up_proj_name="w3",
            ),
            num_experts=int(self.config.n_routed_experts),
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        loaded: set[str] = set()
        suffixes_by_stage = {
            stage_id: set() for stage_id in range(self.model.num_stages)
        }
        raw_attention = {stage_id: {} for stage_id in range(self.model.num_stages)}

        for raw_name, loaded_weight in weights:
            match = _DSPARK_WEIGHT_RE.match(raw_name)
            if match is None:
                continue
            stage_id = int(match.group(1))
            suffix = match.group(2)
            if stage_id >= self.model.num_stages:
                continue
            suffixes_by_stage[stage_id].add(suffix)
            if suffix.startswith("attn."):
                attention_suffix = suffix.removeprefix("attn.")
                if attention_suffix in _ATTENTION_TENSORS:
                    raw_attention[stage_id][attention_suffix] = loaded_weight

            name = self._map_checkpoint_name(raw_name)
            if name is None:
                continue
            if (
                name.endswith("attn.wo_a.weight")
                and loaded_weight.dtype != torch.float8_e4m3fn
                and params.get(name) is not None
                and params[name].dtype == torch.float8_e4m3fn
            ):
                qweight, scale_inv = DeepseekV4ForCausalLM._block_quant_fp8_weight(
                    loaded_weight
                )
                scale_name = name.replace(".weight", ".weight_scale_inv")
                scale_param = params[scale_name]
                scale_param.weight_loader(scale_param, scale_inv)
                loaded.add(scale_name)
                loaded_weight = qweight
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name or ".experts." in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = params.get(mapped_name)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped_name)
                break
            else:
                if moe_loader.matches(name):
                    loaded.add(moe_loader.load(name, loaded_weight))
                    continue
                param = params.get(name)
                if param is None:
                    logger.debug("Skipping unmatched DSpark weight: %s", name)
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded.add(name)

        missing_contract: dict[int, list[str]] = {}
        for stage_id, seen in suffixes_by_stage.items():
            required = set(_ATTENTION_CHECKPOINT_TENSORS | _STAGE_COMMON_CORE)
            if stage_id == 0:
                required.update(_STAGE_ZERO_CORE)
            if stage_id == self.model.num_stages - 1:
                required.update(_LAST_STAGE_CORE)
            for tensor_name, tensor in raw_attention[stage_id].items():
                if (
                    tensor_name.endswith(".weight")
                    and tensor.dtype != torch.float8_e4m3fn
                ):
                    required.discard(f"attn.{tensor_name[:-7]}.scale")
            missing = sorted(required - seen)
            expected_expert_tensors = {
                f"ffn.experts.{expert_id}.{projection}.{tensor_kind}"
                for expert_id in range(int(self.config.n_routed_experts))
                for projection in ("w1", "w2", "w3")
                for tensor_kind in ("scale", "weight")
            }
            missing.extend(sorted(expected_expert_tensors - seen))
            if missing:
                missing_contract[stage_id] = missing
        if missing_contract:
            raise ValueError(
                "DSpark checkpoint is missing required stage weights: "
                f"{missing_contract}."
            )

        shared_parameter_names = {
            "model.embed_tokens.weight",
            "lm_head.weight",
        }
        missing_parameters = sorted(
            name
            for name in params
            if name not in loaded
            and name not in shared_parameter_names
            and not _is_zero_initialized_expert_bias(name)
        )
        if missing_parameters:
            raise ValueError(
                "DSpark checkpoint did not initialize all draft parameters: "
                f"{missing_parameters}."
            )

        for stage_id, source in raw_attention.items():
            device = self.model.stages[stage_id].block.attn_norm.weight.device

            def dequant(name: str) -> torch.Tensor:
                weight = source[f"{name}.weight"].to(device)
                if weight.dtype != torch.float8_e4m3fn:
                    return weight.to(torch.bfloat16)
                return _block_dequant(
                    weight,
                    source[f"{name}.scale"].to(device),
                )

            self.model.stages[stage_id]._dspark_attn = {
                "wq_a": dequant("wq_a"),
                "q_norm_w": source["q_norm.weight"].to(device).to(torch.bfloat16),
                "wq_b": dequant("wq_b"),
                "wkv": dequant("wkv"),
                "kv_norm_w": source["kv_norm.weight"].to(device).to(torch.bfloat16),
                "wo_a": dequant("wo_a"),
                "wo_b": dequant("wo_b"),
                "attn_sink": source["attn_sink"].to(device).float(),
            }

        self.post_load_weights()
        return loaded

    def post_load_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, DeepseekV4Compressor):
                module.process_weights_after_loading()
            elif isinstance(module, DeepseekV4MegaMoEExperts):
                module.finalize_weights()
            elif isinstance(module, MoELayer):
                module.process_weights_after_loading(module)


EntryClass = [DeepseekV4ForCausalLMDSpark]
