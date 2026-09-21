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

"""Checkpoint-local V4.1 DSpark, using the existing fixed-block drafter.

The target supplies HC-mean layer inputs. Three text-only stages share its
embedding/head but have their own 128-expert MoEs and shared experts. Single-pass
HC and FP8 projections use the target implementation; draft attention is dense
within its small context window plus the entire non-causal proposal block, and
runs through the same ``selected_attention`` workspace kernel as the target's
prefill (FlashMLA on sm90+, Triton elsewhere); the fp32 arithmetic stays as the
CPU reference.
"""

from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import rope_inplace
from torch import nn

from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_PREFILL_QUERY_TILE,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.deepseek_v4_dspark import DeepseekV4DSparkModel
from tokenspeed.runtime.models.deepseek_v4_dspark_ops.heads import DSparkVanillaMarkov
from tokenspeed.runtime.models.deepseek_v41 import (
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
    _merged,
    _norm,
    _replicated,
    v41_hc_pre,
    v41_mxfp8_config,
    v41_quantize_fp8,
)
from tokenspeed.runtime.models.target_capture import TargetCaptureConfigurator
from tokenspeed.runtime.utils import add_prefix


def _quantized_kv(x: torch.Tensor) -> torch.Tensor:
    """Materialize the reference all-channel FP8/E8M0 KV round trip as BF16."""
    codes, scales = v41_quantize_fp8(x.reshape(-1, x.shape[-1]))
    values = codes.float().unflatten(-1, (-1, 32))
    values = values * scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return values.flatten(-2).reshape_as(x).to(x.dtype)


@dataclass(frozen=True)
class _WindowSelection:
    """Stage-invariant addressing of one draft forward's window attention.

    ``page``/``row`` gather the history rows of a paged window field by SWA
    slot (negative slots resolve to the null page's first row and are masked
    through ``indices``). ``indices`` is the int32 ``[batch * block, width]``
    selection into the compact per-request workspace ``[history rows, block
    rows]``: history columns are -1 where the slot is invalid, the block's own
    rows are always visible. All three come from ``dsv41.dspark_block`` and
    are shared by every stage.
    """

    page: torch.Tensor
    row: torch.Tensor
    indices: torch.Tensor


def _write_window_rows(
    window: torch.Tensor, slots: torch.Tensor, values: torch.Tensor
) -> None:
    """Scatter rows into a paged window field, leaving invalid slots untouched.

    Negative slots (padding rows, nonresident positions) resolve to the null
    page, which the cache contract keeps zero; those rows write back the bytes
    already there, so the write stays graph-safe without a data-dependent
    branch and the sentinel page keeps its contents.
    """
    rows_per_page = window.shape[1]
    valid = (slots >= 0).unsqueeze(-1)
    slots = slots.clamp_min(0).long()
    page, row = slots // rows_per_page, slots % rows_per_page
    window[page, row] = torch.where(valid, values.to(window.dtype), window[page, row])


class _WindowAttention:
    """Borrow one stage's LCM window pages for one forward; no state survives."""

    def __init__(
        self, positions, window, history_slots, block_size, selection, request_indices
    ):
        self.meta = SimpleNamespace(
            positions=positions, request_indices=request_indices
        )
        self.window = window
        self.history_slots = history_slots
        self.block_size = block_size
        self.selection: _WindowSelection = selection

    def query_metadata(self, mode):
        return self.meta

    def _workspace(self, swa, positions, swa_rope_cache):
        """Per-request ``[batch, window + block, head_dim]`` BF16 rows.

        History rows are gathered straight from the strided paged field; the
        block rows carry the reference all-channel FP8 round trip, exactly as
        the target writes its own SWA rows.
        """
        batch, block = self.history_slots.shape[0], self.block_size
        history = self.window[self.selection.page, self.selection.row]
        if swa.is_cuda and swa.shape[-1] == 512:
            rows = dsv41.dspark_rows(
                swa,
                None,
                None,
                positions if swa_rope_cache is not None else None,
                swa_rope_cache,
                None,
                None,
                torch.empty_like(swa),
            )
        else:
            if swa_rope_cache is not None:
                swa = rope_inplace(swa.clone(), positions, swa_rope_cache, None)
            rows = _quantized_kv(swa)
        return torch.cat((history, rows.reshape(batch, block, -1)), dim=1)

    def forward_v41(
        self,
        q,
        swa,
        *,
        layer_id,
        positions,
        request_indices,
        forward_mode,
        index_q,
        index_weights,
        attn_sink,
        softmax_scale,
        index_process_group,
        swa_rope_cache,
    ):
        if index_q is not None or index_weights is not None:
            raise ValueError("DSpark window attention has no indexer")
        kv = self._workspace(swa, positions, swa_rope_cache)
        if q.is_cuda and q.shape[-1] == 512:
            return self._kernel_forward(q, kv, attn_sink, softmax_scale)
        return self._reference_forward(q, kv, attn_sink, softmax_scale)

    def _kernel_forward(self, q, kv, attn_sink, softmax_scale):
        """The target's prefill workspace kernel over ``[history, block]`` rows.

        ``q`` arrives 64/128-head padded from ``rope_pad_query`` with a
        matching ``-inf``-padded sink; the caller slices the real heads. No
        paged cache is involved, so the SWA/global arguments are all None.
        """
        return dsv41.selected_attention(
            q,
            None,
            None,
            None,
            None,
            None,
            None,
            attn_sink,
            softmax_scale,
            None,
            V41_PREFILL_QUERY_TILE,
            None,
            kv.view(-1, 1, kv.shape[-1]),
            self.selection.indices,
        )

    def _reference_forward(self, q, kv, attn_sink, softmax_scale):
        """FP32 reference arithmetic for CPU and non-512 head dims."""
        batch, block = self.history_slots.shape[0], self.block_size
        queries = q.reshape(batch, block, q.shape[-2], q.shape[-1])
        scores = torch.einsum("bqhd,bkd->bhqk", queries.float(), kv.float())
        scores *= softmax_scale
        visible = self.history_slots >= 0
        valid = torch.cat((visible, visible.new_ones((batch, block))), dim=1)
        scores.masked_fill_(~valid[:, None, None, :], -torch.inf)
        sink = attn_sink[None, :, None, None].expand(batch, -1, block, 1)
        probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
        out = torch.einsum("bhqk,bkd->bqhd", probs, kv.float())
        return out.reshape_as(q).to(q.dtype)


class DeepseekV41DSparkModel(DeepseekV41Model):
    local_base_logits = DeepseekV4DSparkModel.local_base_logits

    def __init__(self, config, mapping, quant_config, prefix):
        if (
            mapping.attn.dp_size != 1
            or mapping.attn.cp_size != 1
            or mapping.pp_size != 1
        ):
            raise NotImplementedError("V4.1 DSpark requires attention DP=CP=PP=1")
        stages = int(config.dspark_num_stages)
        block_size = int(config.dspark_block_size)
        taps = tuple(config.dspark_target_layer_ids)
        if (
            stages != config.num_nextn_predict_layers
            or stages < 1
            or block_size < 1
            or block_size >= config.sliding_window
            or not taps
            or tuple(sorted(set(taps))) != taps
            or taps[0] < 0
            or taps[-1] >= config.num_hidden_layers
        ):
            raise ValueError("Invalid V4.1 DSpark stage/block/capture configuration")
        if not 0 <= config.dspark_noise_token_id < config.vocab_size:
            raise ValueError("DSpark noise token must belong to the target vocabulary")
        if config.dspark_markov_rank < 1:
            raise ValueError("DSpark requires a positive Markov rank")
        draft = copy(config)
        draft.num_hidden_layers = stages
        draft.n_routed_experts = config.dspark_n_routed_experts
        draft.num_experts_per_tok = config.dspark_num_experts_per_tok

        draft.engram_layer_ids = []
        draft.kv_source_layer_ids = []
        draft.index_source_layer_ids = []
        draft.compress_ratios = [0] * stages
        super().__init__(draft, mapping, quant_config, prefix, False, "replicated")
        self.num_stages = stages
        self.block_size = block_size
        self.target_layer_ids = taps
        self.hidden_size = config.hidden_size
        self.noise_token_id = config.dspark_noise_token_id
        self.window_size = config.sliding_window
        self.attention_params = {"head_dim": config.head_dim}
        self.main_proj = _replicated(
            len(taps) * config.hidden_size,
            config.hidden_size,
            torch.bfloat16,
            v41_mxfp8_config(quant_config),
            add_prefix("main_proj", prefix),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Every stage's wkv shard side by side: the context write needs only
        # the KV columns of each stage's wq_a_wkv, so one GEMM serves all
        # stages without computing their wq_a columns.
        self.context_wkv = _merged(
            config.hidden_size,
            [config.head_dim] * stages,
            torch.bfloat16,
            v41_mxfp8_config(quant_config),
            add_prefix("context_wkv", prefix),
        )
        # The bigram table is replicated so the block sampler gathers rows
        # locally; the projection shares the LM head's vocabulary shards.
        self.markov_embedding = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dspark_markov_rank,
            params_dtype=torch.bfloat16,
            org_num_embeddings=None,
            padding_size=64,
            quant_config=None,
            prefix=add_prefix("markov_embedding", prefix),
        )
        self.markov_projection = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dspark_markov_rank,
            bias=False,
            params_dtype=torch.bfloat16,
            org_num_embeddings=None,
            padding_size=64,
            quant_config=None,
            prefix=add_prefix("markov_projection", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            use_presharded_weights=False,
        )
        self.markov_head = DSparkVanillaMarkov(
            self.markov_embedding, self.markov_projection
        )
        # Static verify-all loads this head but does not use confidence truncation.
        self.confidence_projection = nn.Linear(
            config.hidden_size + config.dspark_markov_rank,
            1,
            bias=False,
            device=self.norm.weight.device,
            dtype=torch.float32,
        )

    def _main_input(self, captured):
        if captured.shape[-1] != len(self.target_layer_ids) * self.hidden_size:
            raise ValueError("V4.1 DSpark capture width does not match target taps")
        projected, _ = self.main_proj(
            captured.reshape(-1, captured.shape[-1]),
            block_scale=None,
            output_dtype=None,
        )
        return _norm(projected, self.main_norm)

    def _main_kv(self, attention, kv, positions):
        """Reference chain for one stage's context rows: kv_norm, RoPE, FP8 round trip."""
        return _quantized_kv(
            attention.rotary_emb(_norm(kv, attention.kv_norm), positions, False)
        )

    def write_context_kv(self, captured_hidden_states, positions, slots, cache_pool):
        """Write every stage's window row for the target rows at ``slots``.

        Rows are position-pure functions of the captured hidden state, so the
        same call seeds prefill chunks and refreshes verify rows; the target's
        SWA slots address them.
        """
        if captured_hidden_states.numel() == 0:
            return
        main_x = self._main_input(captured_hidden_states)
        kv, _ = self.context_wkv(main_x, block_scale=None, output_dtype=None)
        positions = positions.reshape(-1)
        slots = slots.reshape(-1)
        head_dim = self.attention_params["head_dim"]
        for stage, layer in enumerate(self.layers):
            window = cache_pool.dspark_kv(stage)
            stage_kv = kv[:, stage * head_dim : (stage + 1) * head_dim]
            if kv.is_cuda and head_dim == 512:
                layer.attn.rotary_emb._prepare_cache(kv.device)
                dsv41.dspark_rows(
                    stage_kv,
                    layer.attn.kv_norm.weight,
                    layer.attn.kv_norm.variance_epsilon,
                    positions,
                    layer.attn.rotary_emb.cos_sin_cache,
                    window,
                    slots,
                    None,
                )
            else:
                _write_window_rows(
                    window, slots, self._main_kv(layer.attn, stage_kv, positions)
                )

    def forward_backbone(
        self, bonus_token_ids, start_pos, history_slots, cache_pool, ctx
    ):
        """Return normalized [batch, proposals, hidden] for a fixed DSpark block.

        ``history_slots`` addresses positions ``start_pos-127..start_pos`` in
        every stage's window; the anchor row at ``start_pos`` was written as a
        verify row before this call.
        """
        batch = bonus_token_ids.numel()
        # Every stage's window field shares the SWA page geometry, so the
        # history addressing and the workspace selection are built once.
        ids, positions, pre_mix, request_indices, page, row, indices = (
            dsv41.dspark_block(
                bonus_token_ids,
                start_pos,
                history_slots,
                self.noise_token_id,
                cache_pool.dspark_kv(0).shape[1],
                self.config.hc_mult,
                self.block_size,
            )
        )
        selection = _WindowSelection(page, row, indices)
        h = self.embed_tokens(ids)
        h = h[:, None, :].repeat(1, self.config.hc_mult, 1)
        for stage, layer in enumerate(self.layers):
            backend = _WindowAttention(
                positions,
                cache_pool.dspark_kv(stage),
                history_slots,
                self.block_size,
                selection,
                request_indices,
            )
            h, pre_mix = layer(
                h,
                pre_mix,
                positions,
                image_mask=None,
                ctx=replace(ctx, attn_backend=backend),
            )
        return _norm(v41_hc_pre(h, pre_mix), self.norm).reshape(
            batch, self.block_size, -1
        )


class DeepseekV41ForCausalLMDSpark(DeepseekV41ForCausalLM, TargetCaptureConfigurator):
    """Strict mtp-only adapter sharing embedding/head with its V4.1 target."""

    def configure_target(self, target_model, target_config) -> None:
        """Install the checkpoint's target taps before draft execution exists."""
        del target_config
        target_model.set_dspark_layers_to_capture(list(self.model.target_layer_ids))

    def __init__(self, config, mapping, quant_config):
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            is_multimodal_active=False,
            mm_attention_backend=None,
        )

    def resolve_model(self, config, mapping, quant_config, prefix):
        return DeepseekV41DSparkModel(
            config.text_config,
            mapping,
            quant_config,
            add_prefix("model", prefix),
        )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        location = super().get_model_config_for_expert_location(config)
        location.num_layers = config.text_config.dspark_num_stages
        location.num_logical_experts = config.text_config.dspark_n_routed_experts
        return location

    def get_hot_token_id(self):
        return None

    def set_embed_and_head(self, embed, head) -> None:
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        return name.removeprefix("model.").startswith("mtp.")

    def _checkpoint_targets(self):
        targets = super()._checkpoint_targets()
        # These tensors are bound from the target before the drafter is constructed.
        del targets["embed.weight"]
        del targets["head.weight"]
        # context_wkv gathers one shard per stage; load_weights presents each
        # stage's checkpoint wkv tensor under a stage-qualified name.
        for field in ("weight", "scale"):
            merged = targets.pop(f"context_wkv.{field}", None)
            if merged is not None:
                for stage in range(self.model.num_stages):
                    targets[f"context_wkv.{stage}.{field}"] = (merged[0], stage)
        return targets

    def load_weights(self, weights, **kwargs) -> None:
        """Load all local mtp constituents through V4.1's exact FP8/FP4 loader."""

        def mapped():
            for raw_name, tensor in weights:
                name = raw_name.removeprefix("model.")
                match = re.fullmatch(r"mtp\.(\d+)\.(.+)", name)
                if match is None or int(match[1]) >= self.model.num_stages:
                    raise ValueError(
                        f"Unexpected V4.1 DSpark checkpoint tensor: {raw_name}"
                    )
                stage, suffix = int(match[1]), match[2]
                wkv = re.fullmatch(r"attn\.wkv\.(weight|scale)", suffix)
                if wkv is not None:
                    # The context write's merged projection loads the same
                    # shard as this stage's own wq_a_wkv.
                    yield f"context_wkv.{stage}.{wkv[1]}", tensor
                if stage == 0 and suffix.startswith(("main_proj.", "main_norm.")):
                    name = suffix
                elif stage == self.model.num_stages - 1 and suffix == "norm.weight":
                    name = suffix
                elif stage == self.model.num_stages - 1 and suffix in (
                    "markov_head.embed.weight",
                    "markov_head.head.weight",
                    "confidence_head.proj.weight",
                ):
                    name = {
                        "markov_head.embed.weight": "markov_embedding.weight",
                        "markov_head.head.weight": "markov_projection.weight",
                        "confidence_head.proj.weight": "confidence_projection.weight",
                    }[suffix]
                else:
                    name = f"layers.{stage}.{suffix}"
                yield name, tensor

        super().load_weights(mapped(), **kwargs)


EntryClass = DeepseekV41ForCausalLMDSpark
