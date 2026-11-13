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

"""Native dense draft arithmetic over the shared fixed-block execution path.

Target features and recurrent features are explicit inputs. This module owns
no request-slot state, cache locations or proposal lifecycle.
Logits retain the shared head's local vocabulary layout until execution code
performs its normal gather or global argmax.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

# Triton-only Inkling2 recurrences outside the registry; Triton serves every supported vendor.
from tokenspeed_kernel.ops.activation._triton.gated_recurrence import (
    capped_logit_sum,
    gated_tanh_recurrence,
    projected_gated_tanh_recurrence,
)
from tokenspeed_kernel.ops.attention.prologue import HeadNorm
from tokenspeed_kernel.ops.embedding import apply_k_rope

# skinny_mm is not a registered GEMM solution; skinny_mm_supported sends other inputs to rnn_step.
from tokenspeed_kernel.ops.gemm.triton_skinny import (
    SKINNY_MM_MAX_ROWS,
    skinny_mm,
    skinny_mm_supported,
)
from tokenspeed_kernel.ops.layernorm.precision import add_rmsnorm_precision
from tokenspeed_kernel.ops.residual.attention import residual_weighted_sum
from torch import nn

from tokenspeed.runtime.configs.inkling2 import Inkling2ModelConfig, Inkling2MTPConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.attention_residual import PrecisionRMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.rotary_embedding import RotaryEmbedding
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base.input_embedding import DecoderInputEmbedding
from tokenspeed.runtime.models.inkling2 import Inkling2ForConditionalGeneration
from tokenspeed.runtime.models.target_capture import TargetCaptureConfigurator

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


def padded_shard_ids_to_vocab(
    ids: torch.Tensor, padded: int, shard: int, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Map ``rank * padded + column`` indices over vocabulary shards padded to
    ``padded`` columns back to vocabulary ids ``rank * shard + column``.

    Args:
        ids: Integer indices, such as an argmax over the padded shards.
        padded: Padded columns per rank.
        shard: Vocabulary rows per rank, ``<= padded``.
        out: Optional integer destination (it may be wider than ``ids``).

    Returns:
        Vocabulary ids in ``ids``' dtype, or ``out``.
    """
    # Integers, so the multiply-subtract is one exact add with alpha.
    return torch.add(
        ids,
        torch.div(ids, padded, rounding_mode="floor"),
        alpha=shard - padded,
        out=out,
    )


def vocab_ids_to_padded_shards(
    ids: torch.Tensor, padded: int, shard: int
) -> torch.Tensor:
    """The inverse of :func:`padded_shard_ids_to_vocab`."""
    return torch.add(
        ids, torch.div(ids, shard, rounding_mode="floor"), alpha=padded - shard
    )


def _norm(width: int, eps: float) -> PrecisionRMSNorm:
    # The draft's reference norms divide by a compile-time width: a reciprocal multiply.
    return PrecisionRMSNorm(width, eps, False, torch.bfloat16, True, True)


class Inkling2MTPAttention(nn.Module):
    def __init__(
        self, config: Inkling2MTPConfig, mapping: Mapping, layer_id: int
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // mapping.attn.tp_size
        self.num_kv_heads = config.num_key_value_heads // mapping.attn.tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        prefix = f"layers.{layer_id}.self_attn"
        parallel = dict(
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
            **parallel,
        )
        self.kv_proj_ctx = QKVParallelLinear(
            config.hidden_size,
            config.head_dim,
            0,
            config.num_key_value_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.kv_proj_ctx",
            **parallel,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
            **parallel,
        )
        self.q_norm = _norm(config.head_dim, config.rms_norm_eps)
        self.k_norm = _norm(config.head_dim, config.rms_norm_eps)
        self.k_norm_ctx = _norm(config.head_dim, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            True,
            torch.bfloat16,
        )
        self.attn = PagedAttention(
            self.num_heads,
            config.head_dim,
            config.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(
                config.sliding_window_size + config.block_size - 1
                if layer_id in config.local_layer_ids
                else -1
            ),
            rotary_emb=self.rotary_emb,
            qk_norm=HeadNorm(
                self.q_norm.weight,
                self.k_norm.weight,
                0.0,
                config.rms_norm_eps,
                round_before_rope=True,
            ),
        )

    def project_context(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw context K/V before per-head normalization and RoPE."""
        kv, _ = self.kv_proj_ctx(hidden)
        return kv.split([self.kv_size, self.kv_size], dim=-1)

    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        q, k, v = self.project_query_block(hidden)
        output = self.attn(q, k, v, positions, ctx)
        return self.o_proj(output.reshape(hidden.shape[0], -1))[0]

    def project_query_block(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return projected BF16 rows; the prologue owns Q/K norm and RoPE."""
        qkv, _ = self.qkv_proj(hidden)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v


class Inkling2MTPMLP(nn.Module):
    def __init__(
        self, config: Inkling2MTPConfig, mapping: Mapping, layer_id: int
    ) -> None:
        super().__init__()
        parallel = dict(
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"layers.{layer_id}.mlp.gate_up_proj",
            **parallel,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            reduce_results=True,
            prefix=f"layers.{layer_id}.mlp.down_proj",
            **parallel,
        )
        self.act = SiluAndMul(swiglu_limit=None)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden)
        return self.down_proj(self.act(gate_up))[0]


def _add_norm(
    hidden: torch.Tensor, residual: torch.Tensor, norm: PrecisionRMSNorm
) -> tuple[torch.Tensor, torch.Tensor]:
    # Normalize the FP32 sum before its separate BF16 residual store.
    return add_rmsnorm_precision(
        hidden,
        residual,
        norm.weight,
        norm.eps,
        norm.pre_weight,
        norm.output_dtype,
        torch.bfloat16,
        mean_by_reciprocal=norm.mean_by_reciprocal,
    )


class Inkling2MTPLayer(nn.Module):
    def __init__(
        self, config: Inkling2MTPConfig, mapping: Mapping, layer_id: int
    ) -> None:
        super().__init__()
        self.input_layernorm = _norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _norm(config.hidden_size, config.rms_norm_eps)
        self.dflare_norm = _norm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Inkling2MTPAttention(config, mapping, layer_id)
        self.mlp = Inkling2MTPMLP(config, mapping, layer_id)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual, normalized = hidden, self.input_layernorm(hidden)
        else:
            normalized, residual = _add_norm(hidden, residual, self.input_layernorm)
        attention = self.self_attn(normalized, positions, ctx)
        normalized, residual = _add_norm(
            attention, residual, self.post_attention_layernorm
        )
        return self.mlp(normalized), residual


class Inkling2MTPHead(nn.Module):
    """Block-local BF16 recurrence with FP64-derived, BF16 stored head folds."""

    def __init__(self, config: Inkling2MTPConfig) -> None:
        super().__init__()
        h, rank = config.hidden_size, config.markov_rank
        self.markov_rank = rank
        self.valid_vocab_size = config.unpadded_vocab_size
        self.softcap = config.final_logit_softcapping
        self.markov_w1 = nn.Embedding(
            config.padded_vocab_size, rank, dtype=torch.bfloat16
        )
        self.rnn_proj = nn.Linear(
            2 * rank + 2 * h, 3 * rank, bias=True, dtype=torch.bfloat16
        )
        self.hidden_out = nn.Linear(rank, h, bias=True, dtype=torch.bfloat16)
        self.dflare_norm = _norm(h, config.rms_norm_eps)
        self.register_buffer("fold_weight", None, persistent=False)
        self.register_buffer("fold_bias", None, persistent=False)
        self.vocab_start = 0
        self.vocab_count = 0

    @torch.no_grad()
    def refresh_fold(
        self, weight: torch.Tensor, *, vocab_start: int, vocab_count: int
    ) -> None:
        self.vocab_start, self.vocab_count = vocab_start, vocab_count
        self.fold_weight = weight.new_empty(weight.shape[0], self.markov_rank)
        self.fold_bias = weight.new_empty(weight.shape[0])
        for begin in range(0, weight.shape[0], 4096):
            end = min(begin + 4096, weight.shape[0])
            rows = weight[begin:end].double()
            self.fold_weight[begin:end].copy_(rows @ self.hidden_out.weight.double())
            self.fold_bias[begin:end].copy_(rows @ self.hidden_out.bias.double())

    def rnn_step(
        self,
        state: torch.Tensor,
        previous: torch.Tensor,
        hidden: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.rnn_proj(
            torch.cat((state, previous, hidden, features), dim=-1)
        )
        if projected.is_cuda:
            return gated_tanh_recurrence(projected, state)
        gate, candidate, output = projected.chunk(3, dim=-1)
        gate = torch.sigmoid(gate)
        return gate * state + (1.0 - gate) * torch.tanh(candidate), torch.tanh(output)

    def step_invariant_projection(
        self, hidden_states: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor | None:
        """FP32 ``[rows, steps, 3 * rank]`` projection of every step's hidden
        row and the features, bias included, or None where ``rnn_step`` runs.

        ``rnn_proj`` sums ``[state | previous | hidden | features]``; only the
        first two change between steps, so a step adds just their products
        and rounds the whole sum once.
        """
        rows, steps, _ = hidden_states.shape
        inputs = torch.cat(
            (hidden_states, features[:, None].expand(-1, steps, -1)), dim=-1
        ).reshape(rows * steps, -1)
        weight = self.rnn_proj.weight[:, 2 * self.markov_rank :]
        if not skinny_mm_supported(inputs[:1], weight):
            return None
        chunks = [
            skinny_mm(
                inputs[start : start + SKINNY_MM_MAX_ROWS],
                weight,
                self.rnn_proj.bias,
                out_dtype=torch.float32,
            )
            for start in range(0, inputs.shape[0], SKINNY_MM_MAX_ROWS)
        ]
        # One chunk (a small batch) is the projection itself: no copy.
        projected = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return projected.view(rows, steps, -1)

    def step_logits(
        self, base: torch.Tensor, latent: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Corrected, soft-capped local logits written into ``out``.

        Args:
            base: BF16 ``[rows, local_vocab]`` base logits.
            latent: BF16 ``[rows, markov_rank]`` recurrent outputs.
            out: BF16 ``[rows, width]`` destination, ``width >= local_vocab``;
                columns past the valid vocabulary are ``-inf``.

        Returns:
            ``out``.
        """
        correction = torch.addmm(self.fold_bias, latent, self.fold_weight.T)
        valid_cols = max(
            0, min(self.vocab_count, self.valid_vocab_size - self.vocab_start)
        )
        if base.is_cuda:
            return capped_logit_sum(
                base, correction, self.softcap or None, valid_cols, out
            )
        logits = base + correction
        if self.softcap:
            logits = (self.softcap * torch.tanh(logits.float() / self.softcap)).to(
                torch.bfloat16
            )
        out.fill_(-float("inf"))
        out[:, :valid_cols] = logits[:, :valid_cols]
        return out

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        f_rnn: torch.Tensor,
        logits_out: torch.Tensor,
        sampler: Callable[[torch.Tensor, int], torch.Tensor],
        shard_padding: tuple[int, int] | None,
    ) -> torch.Tensor:
        """Sample corrected local logits with the execution owner's sampler.

        Recurrent state starts at zero on each call. Every step's logits land
        in ``logits_out`` (``[rows, width >= local_vocab]``, ``-inf`` past the
        valid vocabulary) before the sampler reads them. A distributed sampler
        returns global vocabulary IDs, or with ``shard_padding = (padded,
        shard)`` its ``rank * padded + column`` indices over the padded local
        rows: the steps then feed them back as they are and the returned
        proposals are vocabulary IDs. No model-owned sampling state is
        retained.
        """
        state = hidden_states.new_zeros(hidden_states.shape[0], self.markov_rank)
        tokens = []
        invariant = self.step_invariant_projection(hidden_states, f_rnn)
        previous = first_prev_tokens
        if shard_padding is not None and invariant is None:
            previous = vocab_ids_to_padded_shards(previous, *shard_padding)
        for step in range(base_logits.shape[1]):
            if invariant is None:
                rows = (
                    previous
                    if shard_padding is None
                    else padded_shard_ids_to_vocab(previous, *shard_padding)
                )
                state, latent = self.rnn_step(
                    state,
                    self.markov_w1(rows.long()),
                    hidden_states[:, step],
                    f_rnn,
                )
            else:
                # The first step reads its vocabulary ids' rows directly; later
                # steps read the sampler's indices over the padded shards.
                state, latent = projected_gated_tanh_recurrence(
                    invariant[:, step],
                    state,
                    previous,
                    self.markov_w1.weight,
                    self.rnn_proj.weight[:, : 2 * self.markov_rank],
                    shard_padding=shard_padding if step else None,
                )
            logits = self.step_logits(base_logits[:, step], latent, logits_out)
            previous = sampler(logits, step)
            tokens.append(previous)
        proposals = torch.stack(tokens, dim=1)
        if shard_padding is not None:
            return padded_shard_ids_to_vocab(
                proposals,
                *shard_padding,
                out=torch.empty_like(proposals, dtype=torch.long),
            )
        return proposals.long()


class Inkling2ForConditionalGenerationMTP(nn.Module, TargetCaptureConfigurator):
    """Native draft components with exact checkpoint and shared-resource loading.

    Forward requires common block-decode metadata with the trained query width.
    The shared drafter supplies cache locations for each layer's context writer.

    Target taps follow the checkpoint's vocabulary: raw partials at the named
    layer entries, then a sentinel for the post-final-norm output. These labels
    never receive the completed-layer ``+1`` conversion. Empty forwards keep one
    empty BF16 tensor per trained tap, so the packed width is preserved, and the
    target also hands the final normalized tap to an attached capture sink.
    """

    fall_back_to_pt_during_load = False
    checkpoint_weight_files = ("mtp.safetensors",)

    def __init__(
        self, config: Inkling2MTPConfig, mapping: Mapping, quant_config: None
    ) -> None:
        super().__init__()
        if not isinstance(config, Inkling2MTPConfig) or quant_config is not None:
            raise ValueError("Native draft requires its independent BF16 config")
        if (
            config.q_bias
            or config.o_bias
            or config.is_causal
            or config.sample_from_anchor
        ):
            raise ValueError(
                "Native draft requires unbiased, non-causal block attention"
            )
        if (
            mapping.pp_size != 1
            or mapping.attn.has_dp
            or mapping.dense.has_dp
            or mapping.attn.kvp_size != 1
            or mapping.attn.qcp_size != 1
        ):
            raise NotImplementedError(
                "Native draft currently requires ordinary TP without PP, attention DP, KVP, or QCP"
            )
        if (
            mapping.attn.tp_group != mapping.dense.tp_group
            or config.num_attention_heads % mapping.attn.tp_size
            or config.num_key_value_heads % mapping.attn.tp_size
            or config.intermediate_size % mapping.dense.tp_size
        ):
            raise ValueError(
                "Draft attention and dense TP must match and divide trained dimensions"
            )
        self.config, self.mapping = config, mapping
        self.block_size = config.block_size
        self.num_context_features = len(config.target_layer_ids)
        self.fuse_weight = nn.Parameter(
            torch.zeros(
                config.num_hidden_layers + 1,
                self.num_context_features,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(
            torch.zeros(config.hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.layers = nn.ModuleList(
            [
                Inkling2MTPLayer(config, mapping, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _norm(config.hidden_size, config.rms_norm_eps)
        self.markov_head = Inkling2MTPHead(config)
        self.register_buffer("fuse_alpha", None, persistent=False)
        self.embed_tokens: DecoderInputEmbedding | None = None
        self.lm_head: ParallelLMHead | None = None

    def configure_target(
        self, target_model: nn.Module, target_config: Inkling2ModelConfig
    ) -> None:
        """Configure target features and borrowed modules before cache construction."""
        # A dummy load format never calls load_weights, which derives the mixtures.
        if self.fuse_alpha is None:
            raise RuntimeError("Draft mixtures require complete checkpoint loading")
        if not isinstance(
            target_model, Inkling2ForConditionalGeneration
        ) or not isinstance(target_config, Inkling2ModelConfig):
            raise TypeError("Native draft capture requires an Inkling2 target")
        if (
            self.config.hidden_size != target_config.hidden_size
            or self.config.target_num_hidden_layers != target_config.num_hidden_layers
            or self.config.padded_vocab_size != target_config.padded_vocab_size
            or self.config.unpadded_vocab_size != target_config.unpadded_vocab_size
        ):
            raise ValueError("Native draft checkpoint does not match target geometry")
        target_model.set_dspark_layers_to_capture(list(self.config.target_layer_ids))
        self.attach_shared_modules(
            embed_tokens=target_model.get_decoder_input_embedding(),
            lm_head=target_model.lm_head,
        )

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        return tuple(self.config.target_layer_ids)

    @property
    def context_dtype(self) -> torch.dtype:
        return self.mask_embed.dtype

    @property
    def context_in_features(self) -> int:
        return self.num_context_features * self.config.hidden_size

    def mix_context(self, targets: torch.Tensor, stage: int) -> torch.Tensor:
        values = targets.reshape(-1, self.num_context_features, self.config.hidden_size)
        mixed = residual_weighted_sum(values.contiguous(), self.fuse_alpha[stage])
        norm = (
            self.markov_head.dflare_norm
            if stage == len(self.layers)
            else self.layers[stage].dflare_norm
        )
        return norm(mixed)

    def compute_f_rnn(self, target_hidden_last: torch.Tensor) -> torch.Tensor:
        return self.mix_context(target_hidden_last, len(self.layers))

    def project_context_kv(
        self, target_hidden: torch.Tensor, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_id].self_attn.project_context(
            self.mix_context(target_hidden, layer_id)
        )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        # Each layer mixes the packed taps itself in write_context_kv.
        return target_hidden

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        pool: CachePool,
        *,
        attn_backend: AttentionBackend | None,
    ) -> None:
        """Project packed target features and store each layer's context K/V.

        Positions and destination slots come from the shared round. Each layer
        owns its mixture, context projection and BF16 key-normalization store.
        """
        for layer_id, layer in enumerate(self.layers):
            attn = layer.self_attn
            k, v = self.project_context_kv(target_hidden, layer_id)
            # Per-head rows of the K half of each projection row, read in place.
            k = attn.k_norm_ctx(k.unflatten(-1, (-1, attn.head_dim))).view_as(k)
            k = apply_k_rope(
                positions,
                k,
                attn.head_dim,
                attn.rotary_emb.cos_sin_cache,
                is_neox=attn.rotary_emb.is_neox_style,
                k_rope_out=None,
                solution=None,
                override=None,
            )
            pool.set_kv_buffer(
                attn.attn,
                cache_locs,
                k.view(-1, attn.num_kv_heads, attn.head_dim),
                v.view(-1, attn.num_kv_heads, attn.head_dim),
            )

    def attach_shared_modules(
        self, *, embed_tokens: DecoderInputEmbedding, lm_head: ParallelLMHead
    ) -> None:
        if (
            not isinstance(embed_tokens, DecoderInputEmbedding)
            or embed_tokens.transform is None
            or embed_tokens.weight.dtype != torch.bfloat16
            or embed_tokens.embedding_dim != self.config.hidden_size
            or embed_tokens.num_embeddings != self.config.padded_vocab_size
        ):
            raise ValueError(
                "Draft anchor requires the target's normalized BF16 embedding"
            )
        if (
            lm_head.quant_config is not None
            or lm_head.bias is not None
            or lm_head.weight.dtype != torch.bfloat16
            or lm_head.embedding_dim != self.config.hidden_size
            or lm_head.org_vocab_size != self.config.padded_vocab_size
            or lm_head.num_added_embeddings
            or lm_head.tp_group != self.mapping.lm_head.tp_group
            or lm_head.tp_rank != self.mapping.lm_head.tp_rank
            or lm_head.tp_size != self.mapping.lm_head.tp_size
        ):
            raise ValueError(
                "Draft requires the matching unquantized target LM head layout"
            )
        self.embed_tokens, self.lm_head = embed_tokens, lm_head
        self.refresh_shared_modules()

    def refresh_shared_modules(self) -> None:
        self.markov_head.refresh_fold(
            self.lm_head.weight,
            vocab_start=self.lm_head.shard_indices.org_vocab_start_index,
            vocab_count=self.lm_head.shard_indices.num_org_elements,
        )

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.reshape(-1, self.block_size)
        hidden = self.mask_embed.expand(ids.shape[0], self.block_size, -1).clone()
        hidden[:, 0] = self.embed_tokens(ids[:, 0])
        return hidden.reshape(-1, self.config.hidden_size)

    def compute_base_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return uncapped BF16 shard-local logits, not the target's FP32 logits.

        The proposal rows are a strided view of each block; one GEMM over
        their contiguous copy reads the head once, where a batched GEMM over
        the view would read it once per request.
        """
        rows = hidden.reshape(-1, hidden.shape[-1])
        return F.linear(rows, self.lm_head.weight).view(*hidden.shape[:-1], -1)

    def proposal_rows(self, block_hidden: torch.Tensor) -> torch.Tensor:
        """Select rows 1..Q-1, not the anchor row consumed by vanilla DSpark."""
        return block_hidden[:, 1:]

    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: torch.Tensor | None,
    ) -> LogitsProcessorOutput:
        unsupported = kwargs.keys() - {"input_embeds"}
        if unsupported:
            raise TypeError(
                f"Unsupported native draft arguments: {sorted(unsupported)}"
            )
        hidden = kwargs.get("input_embeds")
        if hidden is None:
            hidden = (
                self.forward_embed(input_ids)
                if input_ids.numel()
                else self.mask_embed.new_empty(0, self.config.hidden_size)
            )
        if not input_ids.numel():
            return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions, ctx)
        hidden = (
            self.norm(hidden)
            if residual is None
            else _add_norm(hidden, residual, self.norm)[0]
        )
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    def _global_shape(self, name: str, param: nn.Parameter) -> tuple[int, ...]:
        h, d, f = (
            self.config.hidden_size,
            self.config.head_dim,
            self.config.intermediate_size,
        )
        q, kv = self.config.num_attention_heads * d, self.config.num_key_value_heads * d
        for suffix, shape in (
            (".qkv_proj.weight", (q + 2 * kv, h)),
            (".kv_proj_ctx.weight", (2 * kv, h)),
            (".o_proj.weight", (h, q)),
            (".gate_up_proj.weight", (2 * f, h)),
            (".down_proj.weight", (h, f)),
        ):
            if name.endswith(suffix):
                return shape
        return tuple(param.shape)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith("lm_head.")
        }
        mapping = (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".k_proj_ctx.", ".kv_proj_ctx.", "k"),
            (".v_proj_ctx.", ".kv_proj_ctx.", "v"),
            (".gate_proj.", ".gate_up_proj.", 0),
            (".up_proj.", ".gate_up_proj.", 1),
        )
        loaded, shards = set(), {}
        self.fuse_alpha = None
        for original, weight in weights:
            name, shard = original, None
            for source, destination, piece in mapping:
                if source in name:
                    name, shard = name.replace(source, destination), piece
                    break
            if name not in params:
                raise ValueError(f"Unrecognized draft checkpoint tensor: {original}")
            param = params[name]
            if weight.dtype != torch.bfloat16 or param.dtype != torch.bfloat16:
                raise ValueError(f"Draft checkpoint must preserve BF16: {original}")
            shape = self._global_shape(name, param)
            if shard is not None:
                rows = (
                    self.config.num_attention_heads * self.config.head_dim
                    if shard == "q"
                    else (
                        self.config.num_key_value_heads * self.config.head_dim
                        if shard in ("k", "v")
                        else self.config.intermediate_size
                    )
                )
                shape = (rows, self.config.hidden_size)
            if tuple(weight.shape) != shape:
                raise ValueError(
                    f"Draft checkpoint shape mismatch for {original}: "
                    f"expected {shape}, got {tuple(weight.shape)}"
                )
            if name in loaded and (shard is None or name not in shards):
                raise ValueError(
                    f"Duplicate or mixed packed/unpacked draft tensor: {original}"
                )
            if shard is not None:
                pieces = shards.setdefault(name, set())
                if shard in pieces:
                    raise ValueError(f"Duplicate draft checkpoint tensor: {original}")
                param.weight_loader(param, weight, shard)
                pieces.add(shard)
            elif name.startswith("layers.") and name.endswith(
                (
                    ".o_proj.weight",
                    ".down_proj.weight",
                    ".qkv_proj.weight",
                    ".kv_proj_ctx.weight",
                    ".gate_up_proj.weight",
                )
            ):
                param.weight_loader(param, weight)
            else:
                default_weight_loader(param, weight)
            loaded.add(name)
        missing = set(params) - loaded
        for name, pieces in shards.items():
            expected = (
                {"q", "k", "v"}
                if ".qkv_proj." in name
                else {"k", "v"} if ".kv_proj_ctx." in name else {0, 1}
            )
            if pieces != expected:
                missing.add(f"{name}: incomplete projection shards")
        if missing:
            raise ValueError(f"Missing draft checkpoint tensors: {sorted(missing)}")
        self.fuse_alpha = torch.softmax(self.fuse_weight.float(), dim=-1).to(
            torch.bfloat16
        )
        if self.lm_head is not None:
            self.refresh_shared_modules()
        return loaded


EntryClass = Inkling2ForConditionalGenerationMTP
