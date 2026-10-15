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

"""Dots3-note text inference with compact DSA and sliding-window MLA caches."""

from collections.abc import Iterable
from dataclasses import replace

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention import attn_merge_state
from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk, dsa_prefill_topk
from tokenspeed_kernel.ops.attention.dsa.triton import workspace_topk_to_global_slots
from tokenspeed_kernel.ops.attention.mla import mla_project_value
from tokenspeed_kernel.ops.attention.prologue import MLAExpandedKV
from torch import nn

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.page_table import (
    build_prefill_kv_workspace_slots,
)
from tokenspeed.runtime.layers.layernorm import LayerNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    get_layer_id,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict


class Dots3NoteIndexer(nn.Module):
    def __init__(self, config, rotary_emb, quant_config, prefix):
        super().__init__()
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.scaling = (self.head_dim * self.num_heads) ** -0.5
        self.rotary_emb = rotary_emb
        for name, input_dim, output_dim in (
            ("wq_b", config.q_lora_rank, self.num_heads * self.head_dim),
            ("wk", config.hidden_size, self.head_dim),
            ("weights_proj", config.hidden_size, self.num_heads),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_dim,
                    output_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix(name, prefix),
                ),
            )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)

    def forward(self, hidden_states, q_latent, positions):
        q = self.wq_b(q_latent)[0].view(-1, self.num_heads, self.head_dim)
        k = self.k_norm(self.wk(hidden_states)[0])
        q_rope, k_rope = self.rotary_emb(
            positions, q[..., : self.rope_dim], k[:, None, : self.rope_dim]
        )
        q[..., : self.rope_dim] = q_rope
        k[:, : self.rope_dim] = k_rope.squeeze(1)
        return q, k, self.weights_proj(hidden_states)[0]


class Dots3NoteAttention(nn.Module):
    def __init__(self, config, mapping, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.layer_id = layer_id
        self.is_swa = config.layer_types[layer_id] == "sliding_attention"
        field_prefix = "swa_" if self.is_swa else ""
        geometry = config.to_dict()
        self.q_lora_rank = geometry[field_prefix + "q_lora_rank"]
        self.kv_lora_rank = geometry[field_prefix + "kv_lora_rank"]
        self.qk_nope_head_dim = geometry[field_prefix + "qk_nope_head_dim"]
        self.qk_rope_head_dim = geometry[field_prefix + "qk_rope_head_dim"]
        self.v_head_dim = geometry[field_prefix + "v_head_dim"]
        self.num_heads = getattr(config, field_prefix + "num_attention_heads")
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.num_heads % mapping.attn.tp_size:
            raise ValueError("dots3-note attention heads must divide evenly over TP")
        self.num_local_heads = self.num_heads // mapping.attn.tp_size
        self.head_start = mapping.attn.tp_rank * self.num_local_heads
        self.q_lora_scale = (config.hidden_size / self.q_lora_rank) ** 0.5
        self.kv_lora_scale = (config.hidden_size / self.kv_lora_rank) ** 0.5
        self.scaling = self.qk_head_dim**-0.5
        self.index_topk = config.index_topk

        replicated = dict(bias=False, quant_config=quant_config)
        self.q_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.q_lora_rank,
            prefix=add_prefix("q_a_proj", prefix),
            **replicated,
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            **replicated,
        )
        # The small gate has fewer than one FP8 block per TP rank.
        self.g_proj = ReplicatedLinear(
            config.hidden_size,
            self.num_heads,
            prefix=add_prefix("g_proj", prefix),
            **replicated,
        )
        parallel = dict(
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_b_proj", prefix),
            **parallel,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            **parallel,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            **parallel,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.k_rope_only_layernorm = RMSNorm(
            self.qk_rope_head_dim, eps=config.rms_norm_eps
        )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, field_prefix + "rope_theta"),
            rope_scaling=None,
            is_neox_style=False,
        )
        window = config.sliding_window_size - 1 if self.is_swa else -1
        self.attn_mqa = PagedAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            sliding_window_size=window,
            rotary_emb=self.rotary_emb,
            qk_norm=None,
        )
        self.attn_mha = PagedAttention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            sliding_window_size=window,
            rotary_emb=self.rotary_emb,
            qk_norm=None,
        )
        self.indexer = (
            None
            if self.is_swa
            else Dots3NoteIndexer(
                config, self.rotary_emb, quant_config, add_prefix("indexer", prefix)
            )
        )
        self.w_kc = self.w_vc = None

    def normalize_latents(self, q, kv):
        q = self.q_a_layernorm(q) * self.q_lora_scale
        latent, rope = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        latent = self.kv_a_layernorm(latent.contiguous()) * self.kv_lora_scale
        rope = self.k_rope_only_layernorm(rope.contiguous())
        return q, torch.cat((latent, rope), dim=-1)

    def _absorb(self, q, latent, positions, ctx, slots):
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        query = torch.empty(
            q.shape[0],
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        torch.bmm(
            q[..., : self.qk_nope_head_dim].transpose(0, 1),
            self.w_kc,
            out=query[..., : self.kv_lora_rank].transpose(0, 1),
        )
        return self.attn_mqa.latent_prologue(
            query,
            q[..., self.qk_nope_head_dim :],
            latent,
            positions,
            ctx,
            slots=slots,
            expanded=None,
            key_rows=None,
        ).query

    def _expanded_kv(self, latent, rope):
        kv = self.kv_b_proj(latent.contiguous())[0].view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value = kv.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        return (
            torch.cat((k_nope, rope.expand(-1, self.num_local_heads, -1)), dim=-1),
            value,
        )

    def _swa_prefill(self, q, latent, positions, ctx, slots, leaf):
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        kv = self.kv_b_proj(latent[:, : self.kv_lora_rank].contiguous())[0].view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value = kv.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        prologue = self.attn_mha.latent_prologue(
            q,
            q[..., self.qk_nope_head_dim :],
            latent,
            positions,
            ctx,
            slots=slots,
            expanded=MLAExpandedKV(k_nope=k_nope, value=value),
            key_rows=None,
        )
        meta = leaf.chunked_prefill_metadata
        output, lse = leaf.forward_extend_chunked(
            prologue.query,
            prologue.key,
            prologue.value,
            self.scaling,
            cum_seq_lens_q=meta.cum_extend_seq_lens,
            cum_seq_lens_kv=meta.cum_extend_seq_lens,
            max_q_len=meta.max_extend_seq_len,
            max_kv_len=meta.max_extend_seq_len,
            seq_lens=meta.extend_seq_lens,
            batch_size=ctx.num_extends,
            causal=True,
        )
        for i in range(meta.chunked_loop_num):
            cached, rope = ctx.token_to_kv_pool.get_mla_kv_buffer(
                self.attn_mha, meta.chunk_kv_indices_list[i], torch.bfloat16
            )
            key, value = self._expanded_kv(cached.squeeze(1), rope)
            part, part_lse = leaf.forward_extend_chunked(
                prologue.query,
                key,
                value,
                self.scaling,
                cum_seq_lens_q=meta.cum_extend_seq_lens,
                cum_seq_lens_kv=meta.cu_chunked_seq_len[i],
                max_q_len=meta.max_extend_seq_len,
                max_kv_len=meta.max_chunk_len_per_loop[i],
                seq_lens=meta.chunked_seq_len[i],
                batch_size=ctx.num_extends,
                causal=False,
                query_start_offsets=meta.prefix_query_start_offsets[i],
            )
            attn_merge_state(output, lse, part, part_lse, inplace=True)
        return output

    def _dsa_prefill(self, query, index_q, weights, ctx, leaf):
        meta = leaf.chunked_prefill_metadata
        prefix = torch.as_tensor(
            meta.extend_prefix_lens_cpu, device="cpu", dtype=torch.int64
        )
        lengths = torch.as_tensor(
            meta.extend_seq_lens_cpu, device="cpu", dtype=torch.int64
        )
        seq_lens_cpu = prefix + lengths
        seq_lens = seq_lens_cpu.to(device=query.device, dtype=torch.int32)
        max_len = int(seq_lens_cpu.max())
        page_size = leaf.kernel_page_size
        table = meta.page_table[:, : (max_len + page_size - 1) // page_size]
        slots = build_prefill_kv_workspace_slots(
            page_table=table,
            seq_lens=seq_lens,
            max_seq_len=max_len,
            page_size=page_size,
            device=query.device,
            num_tokens=int(seq_lens_cpu.sum()),
        )
        requests = torch.repeat_interleave(torch.arange(lengths.numel()), lengths)
        seq_starts = seq_lens_cpu.cumsum(0) - seq_lens_cpu
        query_starts = lengths.cumsum(0) - lengths
        candidates = (
            prefix[requests] + torch.arange(query.shape[0]) - query_starts[requests] + 1
        )
        starts = seq_starts[requests].to(device=query.device, dtype=torch.int32)
        ends = starts + candidates.to(device=query.device, dtype=torch.int32)
        indices, counts = dsa_prefill_topk(
            index_q.contiguous(),
            weights,
            slots,
            starts,
            ends,
            topk=self.index_topk,
            softmax_scale=self.indexer.scaling,
            batch_invariant=False,
            index_k_cache=ctx.token_to_kv_pool.get_index_k_buffer(self.layer_id),
            page_size=page_size,
            max_logits_bytes=global_server_args_dict[
                "deepseek_v4_indexer_prefill_max_logits_mb"
            ]
            * 1024
            * 1024,
            candidate_lens_cpu=candidates,
            slot_order=global_server_args_dict["dsa_slot_order"],
        )
        return leaf.forward_sparse_prefill(
            q=query,
            layer=self.attn_mqa,
            token_to_kv_pool=ctx.token_to_kv_pool,
            kv_seq_lens=ends - starts,
            topk_slots=workspace_topk_to_global_slots(
                workspace_indices=indices, kv_workspace_slots=slots
            ),
            topk_lens=counts,
            max_seq_len=max_len,
        )

    def _dsa_decode_topk(self, index_q, weights, ctx, leaf):
        meta = leaf.forward_decode_metadata
        num_extends = ctx.num_extends
        seq_lens = meta.seq_lens_k[num_extends : ctx.bs]
        table = meta.page_table[num_extends : ctx.bs]
        q_len = index_q.shape[0] // (ctx.bs - num_extends)
        return dsa_decode_topk(
            index_q,
            weights,
            seq_lens,
            table,
            page_size=leaf.kernel_page_size,
            topk=self.index_topk,
            softmax_scale=self.indexer.scaling,
            batch_invariant=False,
            q_len_per_req=q_len,
            index_k_cache=ctx.token_to_kv_pool.get_index_k_buffer(self.layer_id),
            seq_lens_2d=meta._dsa_seq_lens_2d[num_extends * q_len :],
            plan=meta._dsa_plan,
            slot_order=global_server_args_dict["dsa_slot_order"],
        )

    @break_point
    def forward(self, positions, hidden_states, ctx, comm_manager, block_scale=None):
        if hidden_states.shape[0] == 0:
            return hidden_states
        with ctx.attn_backend.record_pd_cache_step(
            ctx.forward_mode, writes_in_call=True, record_kv_cache=None
        ):
            return self._forward(
                positions, hidden_states, ctx, comm_manager, block_scale
            )

    def _forward(self, positions, hidden_states, ctx, comm_manager, block_scale):
        hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
        q_latent, latent = self.normalize_latents(
            self.q_a_proj(hidden_states)[0], self.kv_a_proj_with_mqa(hidden_states)[0]
        )
        q = self.q_b_proj(q_latent)[0]
        backend = ctx.attn_backend
        leaf = backend.leaves[self.attn_mqa.group_id]
        num_prefill = (
            int(leaf.chunked_prefill_metadata.extend_seq_lens_cpu.sum())
            if ctx.num_extends
            else 0
        )
        slots = backend.forward_write_locations(self.attn_mqa, ctx.forward_mode)
        output = torch.empty(
            q.shape[0],
            self.num_local_heads,
            self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        if self.is_swa:
            if num_prefill:
                output[:num_prefill] = self._swa_prefill(
                    q[:num_prefill],
                    latent[:num_prefill],
                    positions[:num_prefill],
                    ctx,
                    slots[:num_prefill],
                    leaf,
                )
            query = (
                self._absorb(
                    q[num_prefill:],
                    latent[num_prefill:],
                    positions[num_prefill:],
                    ctx,
                    slots[num_prefill:],
                )
                if num_prefill < q.shape[0]
                else None
            )
        else:
            index_q, index_k, weights = self.indexer(hidden_states, q_latent, positions)
            ctx.token_to_kv_pool.set_index_k_buffer(
                self.layer_id, slots, index_k, write_mask=None
            )
            query = self._absorb(q, latent, positions, ctx, slots)
            if num_prefill:
                attended = self._dsa_prefill(
                    query[:num_prefill],
                    index_q[:num_prefill],
                    weights[:num_prefill],
                    ctx,
                    leaf,
                )
                mla_project_value(
                    attended.view(-1, self.num_local_heads, self.kv_lora_rank),
                    self.w_vc,
                    out=output[:num_prefill].flatten(1),
                )
            query = query[num_prefill:]
        if num_prefill < q.shape[0]:
            kwargs = {}
            if not self.is_swa:
                topk, counts = self._dsa_decode_topk(
                    index_q[num_prefill:], weights[num_prefill:], ctx, leaf
                )
                kwargs.update(topk_indices=topk, topk_lens=counts)
            decode_ctx = replace(
                ctx,
                forward_mode=ForwardMode.DECODE,
                num_extends=0,
                bs=ctx.bs - ctx.num_extends,
            )
            attended = self.attn_mqa(
                query, None, None, positions=None, ctx=decode_ctx, **kwargs
            )
            mla_project_value(
                attended.view(-1, self.num_local_heads, self.kv_lora_rank),
                self.w_vc,
                out=output[num_prefill:].flatten(1),
            )
        gate = self.g_proj(hidden_states)[0][
            :, self.head_start : self.head_start + self.num_local_heads
        ]
        output.mul_(gate.sigmoid().unsqueeze(-1))
        return self.o_proj(output.flatten(1))[0]


class Dots3NoteRouter(nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.weight = gate.weight
        # Keep the parameter shared with the already configured TopK / MoE layer.
        self.e_score_correction_bias = gate.e_score_correction_bias

    def forward(self, hidden_states):
        return F.linear(hidden_states.float(), self.weight.float())


class Dots3NoteDecoderLayer(DeepseekV3DecoderLayer):
    @property
    def attention_cls(self):
        return Dots3NoteAttention

    def __init__(
        self, config, layer_id, mapping, quant_config=None, prefix="", alt_stream=None
    ):
        nn.Module.__init__(self)
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.self_attn = self.attention_cls(
            config, mapping, layer_id, quant_config, add_prefix("self_attn", prefix)
        )
        self.is_moe_layer = self._is_moe_layer(layer_id, False, config)
        if self.is_moe_layer:
            self.mlp = DeepseekV3MoE(
                config,
                mapping,
                quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
            self.mlp.gate = Dots3NoteRouter(self.mlp.gate)
        else:
            self.mlp = DeepseekV3MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                mapping,
                quant_config,
                prefix=add_prefix("mlp", prefix),
                batch_invariant=False,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=self._is_moe_layer(layer_id - 1, False, config),
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            dense_batch_invariant=False,
            query_sharded=False,
        )


class Dots3NoteModel(DeepseekV3Model):
    def __init__(self, config, mapping, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.mapping = mapping
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                Dots3NoteDecoderLayer(
                    config,
                    i,
                    mapping,
                    quant_config,
                    add_prefix(f"layers.{i}", prefix),
                    alt_stream=self.alt_stream,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = set()
        self._dflash_capture_idx_map = {}


class Dots3NoteForCausalLM(DeepseekV3ForCausalLM):
    model_cls = Dots3NoteModel

    def __init__(
        self,
        config,
        mapping,
        quant_config=None,
        prefix="",
        *,
        is_multimodal_active=False,
        mm_attention_backend=None,
    ):
        if is_multimodal_active:
            raise NotImplementedError(
                "dots3-note encoders are not implemented; use --language-model-only"
            )
        if (
            mapping.attn.has_head_tp
            or global_server_args_dict["tp_batch_invariant"] != "none"
        ):
            raise NotImplementedError(
                "dots3-note does not yet support attention head TP"
            )
        if (
            not config.apply_mla_qkv_lora_rescale
            or config.attention_gate_type != "headwise"
            or config.swa_attention_gate_type != "headwise"
            or config.attention_bias
            or config.rope_scaling is not None
        ):
            raise ValueError("Unsupported dots3-note attention configuration")
        super().__init__(config, mapping, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        loaded = set()
        for name, tensor in weights:
            layer_id = get_layer_id(name)
            # Text-only phase: the checkpoint also carries encoders and one MTP head.
            if name.startswith(("vision_encoder.", "audio_encoder.", "model.mtp.")) or (
                layer_id is not None and layer_id >= self.config.num_hidden_layers
            ):
                continue
            if moe_loader.matches(name):
                loaded.add(moe_loader.load(name, tensor))
                continue
            if moe_loader.is_expert_checkpoint_weight(name):
                continue
            shard = None
            for source, index in (("gate_proj", 0), ("up_proj", 1)):
                if f".{source}." in name:
                    name = name.replace(f".{source}.", ".gate_up_proj.")
                    shard = index
                    break
            if name not in params:
                raise ValueError(f"Unexpected dots3-note text weight: {name}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, tensor)
            else:
                loader(param, tensor, shard)
            loaded.add(name)
        self.post_load_weights()
        return loaded


EntryClass = [Dots3NoteForCausalLM]
