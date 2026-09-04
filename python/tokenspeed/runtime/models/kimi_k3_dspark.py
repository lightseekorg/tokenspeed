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
#
# The architecture follows the reference DSpark MLA draft model in vLLM
# (vllm/models/kimi_k3/nvidia/dspark_mla.py, Apache-2.0, Copyright contributors
# to the vLLM project).

"""MLA-native Kimi-K3 DSpark draft model.

Five dense DeepSeek-V3-style MLA layers that draft a seven-token block in one
non-autoregressive forward. Target hidden states tapped at five K3 layers are
concatenated, projected by ``context_proj``, and written into the draft's own
latent KV cache; the block's queries are one anchor token plus mask tokens, and
the intra-block token dependence comes entirely from the rank-256 Markov head.

The draft borrows the target's embedding and lm_head, so the checkpoint's
``embed_tokens`` copy is skipped and no ``lm_head`` is shipped at all.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from tokenspeed_kernel.ops.embedding import apply_k_rope
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_dspark_config import (
    K3_DSPARK_SKIPPED_WEIGHT_PREFIXES,
    k3_dspark_inactive_features,
    validate_k3_dspark_config,
)
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.segmented_rmsnorm import segmented_rmsnorm
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.models.dflash import DFlashMLP
from tokenspeed.runtime.models.dspark import VanillaMarkov
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger

logger = get_colorful_logger(__name__)


class K3DSparkAttention(DeepseekV3AttentionMLA):
    """RoPE MLA for the draft's non-causal block decode.

    The draft only ever runs one shape: a whole block of queries per request in
    a single decode forward. The base class's prefill/decode split exists to
    serve a target and would mis-route these rows, so ``_attn`` collapses to the
    absorbed decode path. Non-causality is not a property of this module -- the
    MLA backend supplies it by giving every block row the same block-end
    ``seq_len``, so each query sees the whole block including its own future.
    """

    def _attn(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        if q.size(0) == 0:
            return q.new_empty((0, self.num_local_heads * self.v_head_dim))
        if self.w_kc is None or self.w_vc is None:
            # The absorbed decode kernel takes these as raw pointers, so a
            # missing post_load_weights surfaces as a null-address GPU fault
            # rather than anything that names the cause.
            raise RuntimeError(
                "K3 DSpark absorbed MLA factors are missing; "
                "post_load_weights did not run after loading the draft."
            )

        decode_ctx = ctx
        if not ctx.forward_mode.is_decode():
            from dataclasses import replace

            decode_ctx = replace(ctx, forward_mode=ForwardMode.DECODE)

        Q, K = self.forward_absorb_qkv_proj(
            q, latent_cache, positions, decode_ctx, out_cache_loc
        )
        attn_output = q.new_empty((q.size(0), self.num_local_heads * self.v_head_dim))
        self.forward_absorb_attn_v_proj(
            Q, K, decode_ctx, out_cache_loc, attn_output, record_kv_cache=False
        )
        return attn_output

    @torch.no_grad()
    def project_latent_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Latent KV rows for context injection: ``[c_KV_norm | k_PE_raw]``.

        Returns the pre-RoPE latent so the caller can apply positions in one
        batched step. Only the KV half of the fused down-projection is used; the
        Q half is dead work here, which is why the unquantized path slices the
        weight instead of running the full GEMM.
        """
        kv_width = self.kv_lora_rank + self.qk_rope_head_dim
        fused = getattr(self, "fused_qkv_a_proj_with_mqa", None)
        if fused is not None:
            weight = getattr(fused, "weight", None)
            if weight is not None and weight.dtype == hidden_states.dtype:
                latent = torch.nn.functional.linear(
                    hidden_states, weight[self.q_lora_rank :]
                )
            else:
                latent = fused(hidden_states, None, torch.bfloat16)[
                    ..., self.q_lora_rank :
                ]
        else:
            latent = self.kv_a_proj_with_mqa(hidden_states)[0]
        latent = latent[..., :kv_width]
        # Both halves must be contiguous in their own right: the norm kernel
        # rejects a strided output, and a column slice of a [T, 576] row is
        # strided even though each row's bytes are adjacent.
        kv_a = self.kv_a_layernorm(latent[..., : self.kv_lora_rank].contiguous())
        k_pe = latent[..., self.kv_lora_rank :].contiguous()
        return torch.cat((kv_a, k_pe), dim=-1)

    def apply_latent_rope(
        self, positions: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """Rotate the k_PE tail of already-normalized latent rows in place.

        The rope tail is contiguous within each row, so the slice reshapes to a
        *view* of ``latent`` and the rope kernel rotates it in place. Copy it
        out first: writing the result back onto its own storage is a
        self-assignment torch refuses.
        """
        if self.rotary_emb is None or latent.size(0) == 0:
            return latent
        k_pe = (
            latent[..., self.kv_lora_rank :]
            .reshape(-1, 1, self.qk_rope_head_dim)
            .clone()
        )
        k_pe_rot = apply_k_rope(
            positions,
            k_pe,
            self.qk_rope_head_dim,
            self.rotary_emb.cos_sin_cache,
            is_neox=self.rotary_emb.is_neox_style,
        )
        latent[..., self.kv_lora_rank :] = k_pe_rot.reshape(
            latent.size(0), self.qk_rope_head_dim
        )
        return latent


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.mapping = mapping
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = K3DSparkAttention(
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=int(config.num_attention_heads),
            qk_nope_head_dim=int(config.qk_nope_head_dim),
            qk_rope_head_dim=int(config.qk_rope_head_dim),
            v_head_dim=int(config.v_head_dim),
            q_lora_rank=int(config.q_lora_rank),
            kv_lora_rank=int(config.kv_lora_rank),
            rope_theta=config.resolved_rope_theta(),
            rope_scaling=config.rope_scaling_dict(),
            max_position_embeddings=int(config.max_position_embeddings),
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
        )
        # The draft's MLA layers join the target's full_attention cache
        # group (K3 publishes 4 groups: full_attention + 3 KDA linear). The
        # inherited attn_mqa/attn_mha are built without a group_id; tag them so
        # validate_cache_group_ids binds them to the full_attention table
        # instead of failing on an empty group_id. Mirrors the target
        # (kimi_k3.py MLA layer construction).
        self.self_attn.attn_mqa.group_id = FULL_ATTENTION
        self.self_attn.attn_mha.group_id = FULL_ATTENTION
        self.layer_id = layer_id
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = DFlashMLP(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        # o_proj and down_proj both leave their row-parallel output un-reduced;
        # the next norm reduces it, fused where the token count allows.
        self.self_attn.o_proj.reduce_results = False
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=False,
            prev_is_moe=False,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self._norm_with_allreduce(
                self.input_layernorm,
                hidden_states,
                residual,
                ctx,
                self.mapping.dense,
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            comm_manager=self.comm_manager,
        )
        hidden_states, residual = self._norm_with_allreduce(
            self.post_attention_layernorm,
            hidden_states,
            residual,
            ctx,
            self.mapping.attn,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    @staticmethod
    def _norm_with_allreduce(
        norm: RMSNorm,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        group,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce the preceding row-parallel output, then norm.

        Deliberately the unfused path. The fused variant reduces through Iris
        symmetric-heap buffers sized from the target's shapes, and the draft's
        block forward is a different width; until that is verified the plain
        all-reduce is the correct default. It costs one extra pass over an
        8-row tensor.
        """
        del ctx
        hidden_states = all_reduce(hidden_states, group.tp_group)
        return norm(hidden_states, residual)


class K3DSparkModel(nn.Module):
    """The draft network. Interface-compatible with ``DFlashDraftModel``."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        validate_k3_dspark_config(config)
        for note in k3_dspark_inactive_features(config):
            logger.warning("K3 DSpark: %s", note)
        self.config = config
        self.mapping = mapping
        self.attention_kind = "kimi_mla"
        hidden_size = int(config.hidden_size)
        eps = float(config.rms_norm_eps)

        self.num_context_features = len(config.target_layer_ids)
        self.context_proj = ReplicatedLinear(
            self.num_context_features * int(config.target_hidden_size),
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("context_proj", prefix),
        )
        self.context_norm = RMSNorm(hidden_size, eps=eps)
        self.fc_norm = (
            nn.ModuleList(
                [
                    RMSNorm(int(config.target_hidden_size), eps=eps)
                    for _ in range(self.num_context_features)
                ]
            )
            if bool(getattr(config, "fc_norm", False))
            else None
        )
        self.register_buffer("_fc_norm_weight", None, persistent=False)

        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    config=config,
                    mapping=mapping,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(int(config.num_hidden_layers))
            ]
        )
        self.final_norm = RMSNorm(hidden_size, eps=eps)
        self.markov_head = VanillaMarkov(
            vocab_size=int(config.vocab_size),
            markov_rank=int(config.markov_rank),
        )
        # Names the DFlash drafter reads off the draft model.
        self.block_size = None
        self.hidden_size = hidden_size

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Concatenated target taps -> draft hidden space."""
        if self.fc_norm is not None:
            num_segments = len(self.fc_norm)
            hidden_size = int(self.config.target_hidden_size)
            expected_width = num_segments * hidden_size
            if target_hidden.shape[-1] != expected_width:
                raise ValueError(
                    f"fc_norm expects {num_segments} target taps ({expected_width} "
                    f"features), got {target_hidden.shape[-1]} features"
                )
            fc_norm_weight = self._fc_norm_weight
            if fc_norm_weight is None:
                fc_norm_weight = torch.stack(
                    [norm.weight.detach() for norm in self.fc_norm], dim=0
                ).contiguous()
                self._fc_norm_weight = fc_norm_weight
            target_hidden = target_hidden.unflatten(-1, (num_segments, hidden_size))
            target_hidden = segmented_rmsnorm(
                target_hidden,
                fc_norm_weight,
                float(self.fc_norm[0].variance_epsilon),
            )
            target_hidden = target_hidden.flatten(-2)
        return self.context_norm(self.context_proj(target_hidden)[0])

    def _finalize_hidden(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Reduce the last row-parallel MLP output before final normalization."""
        hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states

    # ------------------------------------------------------------------
    # Context-injection contract (see DFlashDraftModel for the GQA counterpart)
    # ------------------------------------------------------------------

    @property
    def context_in_features(self) -> int:
        return self.num_context_features * int(self.config.target_hidden_size)

    @property
    def context_dtype(self) -> torch.dtype:
        return self.context_proj.weight.dtype

    @torch.no_grad()
    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        token_to_kv_pool,
    ) -> None:
        """Project target-derived context into each draft layer's latent cache.

        One row per token per layer, laid out ``[c_KV_norm | k_PE_RoPE]``. The
        Q half of the fused down-projection is dead work here, so
        ``project_latent_kv`` slices the weight instead of running it.
        """
        if ctx_hidden.shape[0] == 0:
            return
        for layer in self.layers:
            attn = layer.self_attn
            latent = attn.project_latent_kv(ctx_hidden)
            latent = attn.apply_latent_rope(positions, latent)
            token_to_kv_pool.set_mla_kv_buffer(
                attn.attn_mqa,
                cache_locs,
                latent[..., : attn.kv_lora_rank].contiguous(),
                latent[..., attn.kv_lora_rank :].contiguous(),
            )

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        kv_sync_event=None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError("K3DSparkModel requires input_embeds.")
            hidden_states = self.context_norm.weight.new_empty(
                (0, int(self.config.hidden_size))
            )
        else:
            hidden_states = input_embeds
        # A DP-idle rank has no draft rows. K3 DSPARK's draft is dense and all
        # of its collectives are scoped to the local attention TP group, whose
        # peers are idle together, so there is no cross-DP collective to join.
        # In particular, FlashInfer's SiLU kernel cannot launch with M=0.
        if hidden_states.shape[0] == 0:
            return LogitsProcessorOutput(
                next_token_logits=None, hidden_states=hidden_states
            )
        # A zero residual rather than None: the drafter hands over an un-reduced
        # embedding, so the first layer must take the all-reduce path too.
        residual = torch.zeros_like(hidden_states)

        if kv_sync_event is not None:
            torch.cuda.current_stream().wait_event(kv_sync_event)

        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                residual=residual,
            )

        if residual is None:
            hidden_states = self.final_norm(hidden_states)
        else:
            hidden_states = self._finalize_hidden(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load the 68-tensor DSpark checkpoint.

        Two shapes need routing beyond the default loader: ``gate_proj``/
        ``up_proj`` stack into ``gate_up_proj``, and ``q_a_proj``/
        ``kv_a_proj_with_mqa`` stack into the fused down-projection at offsets
        0 and ``q_lora_rank``.
        """
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        fused_qkv_a_offsets = {
            "q_a_proj": 0,
            "kv_a_proj_with_mqa": int(self.config.q_lora_rank),
        }
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        unexpected: list[str] = []

        for name, loaded_weight in weights:
            name = name.removeprefix("model.")
            if name.startswith(K3_DSPARK_SKIPPED_WEIGHT_PREFIXES):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                target = name.replace(weight_name, param_name)
                param = params_dict.get(target)
                if param is None:
                    continue
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(target)
                break
            else:
                fused_key = next(
                    (key for key in fused_qkv_a_offsets if f".{key}." in name), None
                )
                if fused_key is not None:
                    target = name.replace(fused_key, "fused_qkv_a_proj_with_mqa")
                    param = params_dict.get(target)
                    if param is None:
                        unexpected.append(name)
                        continue
                    param.weight_loader(
                        param, loaded_weight, begin_size=fused_qkv_a_offsets[fused_key]
                    )
                    loaded.add(target)
                    continue

                param = params_dict.get(name)
                if param is None:
                    unexpected.append(name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        if unexpected:
            raise ValueError(
                f"K3 DSpark checkpoint has {len(unexpected)} unexpected weights "
                f"with no destination: {sorted(unexpected)[:8]}"
            )
        missing = sorted(set(params_dict) - loaded)
        if missing:
            raise ValueError(
                f"K3 DSpark checkpoint is missing {len(missing)} weights: "
                f"{missing[:8]}"
            )
        self.post_load_weights()

    def post_load_weights(self) -> None:
        """Precompute the absorbed decode factors from kv_b_proj."""
        for layer in self.layers:
            self_attn = layer.self_attn
            self_attn.w_kc, self_attn.w_vc = _prepare_mla_kv_b_proj_weights(
                self_attn.kv_b_proj.weight, self_attn
            )
        if self.fc_norm is not None:
            self._fc_norm_weight = torch.stack(
                [norm.weight.detach() for norm in self.fc_norm], dim=0
            ).contiguous()


EntryClass = [K3DSparkModel]
