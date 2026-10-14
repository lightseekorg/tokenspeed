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


"""Single-layer SWA MTP draft using the shared NextN execution path."""

from collections.abc import Iterable
from dataclasses import replace

import torch
from tokenspeed_kernel.ops.attention.mla import mla_project_value
from torch import nn

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from tokenspeed.runtime.models.dots3_note import (
    Dots3NoteAttention,
    Dots3NoteDecoderLayer,
)
from tokenspeed.runtime.utils import add_prefix


class Dots3NoteDraftAttention(Dots3NoteAttention):
    def forward(self, positions, hidden_states, ctx, comm_manager, block_scale=None):
        if ctx.draft_narrowing is None or hidden_states.shape[0] == 0:
            return super().forward(
                positions, hidden_states, ctx, comm_manager, block_scale
            )
        hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
        q_latent, latent = self.normalize_latents(
            self.q_a_proj(hidden_states)[0], self.kv_a_proj_with_mqa(hidden_states)[0]
        )
        q = self.q_b_proj(q_latent)[0]
        ctx.draft_narrowing.publish_accepted_prefix()
        slots = ctx.attn_backend.forward_write_locations(
            self.attn_mqa, ForwardMode.DECODE
        )
        decode_ctx = replace(ctx, forward_mode=ForwardMode.DECODE)
        # Write all verified rows, then attend only from each request's live row.
        query = self._absorb(q, latent, positions, decode_ctx, slots)
        query = query.index_select(0, ctx.gather_ids)
        with ctx.attn_backend.override_num_extends(0):
            attended = self.attn_mqa(
                query,
                None,
                None,
                positions=None,
                ctx=decode_ctx,
                record_kv_cache=not ctx.forward_mode.is_decode_or_idle(),
            )
        output = q.new_empty(ctx.bs, self.num_local_heads, self.v_head_dim)
        mla_project_value(
            attended.view(-1, self.num_local_heads, self.kv_lora_rank),
            self.w_vc,
            out=output.flatten(1),
        )
        gate = self.g_proj(hidden_states.index_select(0, ctx.gather_ids))[0][
            :, self.head_start : self.head_start + self.num_local_heads
        ]
        output.mul_(gate.sigmoid().unsqueeze(-1))
        return self.o_proj(output.flatten(1))[0]


class Dots3NoteDraftDecoderLayer(Dots3NoteDecoderLayer):
    @property
    def attention_cls(self):
        return Dots3NoteDraftAttention

    def narrow_residual(self, residual, ctx):
        if ctx.draft_narrowing is None:
            return residual
        return residual.index_select(0, ctx.gather_ids)


class Dots3NoteModelNextN(nn.Module):
    def __init__(self, config, mapping, quant_config=None, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("eh_proj", prefix),
        )
        self.decoder = Dots3NoteDraftDecoderLayer(
            config, 0, mapping, quant_config, add_prefix("decoder", prefix)
        )
        self.shared_head = nn.ModuleDict(
            {"norm": RMSNorm(config.hidden_size, eps=config.rms_norm_eps)}
        )

    def forward(self, input_ids, positions, ctx, captured_hidden_states=None):
        if ctx.forward_mode.is_idle():
            fused = self.enorm.weight.new_empty((0, 2 * self.hidden_size))
        else:
            if captured_hidden_states is None:
                raise ValueError("dots3-note MTP requires captured_hidden_states")
            fused = torch.cat(
                (
                    self.enorm(self.embed_tokens(input_ids)),
                    self.hnorm(captured_hidden_states),
                ),
                dim=-1,
            )
        hidden_states = self.eh_proj(fused)[0]
        hidden_states, residual = self.decoder(positions, hidden_states, ctx, None)
        if not ctx.forward_mode.is_idle():
            hidden_states, _ = self.decoder.comm_manager.final_norm(
                hidden_states, residual, ctx, self.shared_head["norm"]
            )
        return hidden_states, None


class Dots3NoteForCausalLMNextN(BaseCausalLM):
    model_cls = Dots3NoteModelNextN
    forward = DeepseekV3ForCausalLMNextN.forward
    get_hot_token_id = DeepseekV3ForCausalLMNextN.get_hot_token_id
    post_load_weights = DeepseekV3ForCausalLMNextN.post_load_weights

    def post_init(self):
        self.logits_processor.do_argmax = True

    def set_embed_and_head(self, embed, head):
        # The checkpoint's MTP embedding differs from the target embedding.
        super().set_embed_and_head(self.model.embed_tokens.weight, head)

    def checkpoint_weight_name_filter(self, name):
        return name.startswith(
            (f"model.layers.{self.config.mtp_layer_offset}.", "model.mtp.")
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        loaded = set()
        layer_prefix = f"model.layers.{self.config.mtp_layer_offset}."
        for name, weight in weights:
            if not self.checkpoint_weight_name_filter(name):
                continue
            if name.startswith("model.mtp."):
                name = "model." + name.removeprefix("model.mtp.")
            else:
                suffix = name.removeprefix(layer_prefix)
                prefix = (
                    "model."
                    if suffix.startswith(
                        ("enorm.", "hnorm.", "eh_proj.", "shared_head.norm.")
                    )
                    else "model.decoder."
                )
                name = prefix + suffix
            for shard, shard_id in (("gate_proj", 0), ("up_proj", 1)):
                if f".mlp.{shard}." in name:
                    name = name.replace(f".{shard}.", ".gate_up_proj.")
                    param = params[name]
                    param.weight_loader(param, weight, shard_id)
                    break
            else:
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight)
            loaded.add(name)
        self.post_load_weights()
        return loaded


EntryClass = [Dots3NoteForCausalLMNextN]
