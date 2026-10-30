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

"""Ascend MLA projections, bounded Prefill and prepared native Decode."""

from __future__ import annotations

import weakref

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.attention.mla import (
    mla_prefill,
    prepare_mla_prolog,
    prepare_mla_resources,
)

from tokenspeed.runtime.layers.attention.backends.paged.ascend_mla import (
    AscendMLABackend,
    ascend_mla_prefill_backend,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm
from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLAAttention


class AscendMLAQKNorm(FusedRMSNorm):
    """Use registered RMSNorm leaves instead of the CUDA-only dual-norm call."""

    def forward(self, input_q_a, input_kv_a, output_q_a=None, output_kv_a=None):
        q_out = input_q_a if output_q_a is None else output_q_a
        kv_out = input_kv_a if output_kv_a is None else output_kv_a
        self.q_a_norm(input_q_a, out=q_out)
        self.kv_a_norm(input_kv_a, out=kv_out)
        return q_out, kv_out


class AscendMLAAttention(KimiLinearMLAAttention):
    """Keep empty local SP shards in ungated MLA's projection AllGather."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fused_decode: FusedMLADecode | None = None
        self.optimized_prefill = (
            not self.use_output_gate
            and self.rotary_emb is None
            and self.q_lora_rank is not None
        )
        if self.q_lora_rank is not None:
            self.fused_qk_layernorm = AscendMLAQKNorm(
                self.q_a_layernorm, self.kv_a_layernorm
            )

    def _prefill_sequence(self, query, read_rows, *, prefix, kv_block_tokens):
        """Bound expanded history; merge partial attention in FP32.

        History blocks precede every query and need no causal mask. The
        final block contains all current query rows and is causal. Keep a
        short sequence in one call to preserve its existing arithmetic.
        """
        query_rows = query.shape[0]
        total = prefix + query_rows
        if query_rows < 1 or query_rows > kv_block_tokens or prefix < 0:
            raise ValueError("MLA Prefill needs a nonempty bounded query and prefix")
        if total <= kv_block_tokens:
            spans = [(0, total, True)]
        else:
            spans = [
                (start, min(start + kv_block_tokens, prefix), False)
                for start in range(0, prefix, kv_block_tokens)
            ] + [(prefix, total, True)]
        output = normalizer = None
        cumulative_q = torch.tensor([0, query_rows], dtype=torch.int64)
        for start, end, causal in spans:
            rows = read_rows(start, end)
            expanded = self.kv_b_proj(rows[:, : self.kv_lora_rank].contiguous())[0]
            expanded = expanded.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            key = torch.cat(
                (
                    expanded[..., : self.qk_nope_head_dim],
                    rows[:, None, self.kv_lora_rank :].expand(
                        -1, self.num_local_heads, -1
                    ),
                ),
                dim=-1,
            )
            partial, lse = mla_prefill(
                q=query,
                k=key,
                v=expanded[..., self.qk_nope_head_dim :].contiguous(),
                cu_seqlens_q=cumulative_q,
                cu_seqlens_kv=torch.tensor([0, end - start], dtype=torch.int64),
                max_seqlen_q=query_rows,
                max_seqlen_kv=end - start,
                softmax_scale=self.attn_mha.scaling,
                is_causal=causal,
                return_lse=True,
            )
            if output is None:
                output, normalizer = partial.float(), lse.float()
            else:
                combined = torch.logaddexp(normalizer, lse.float())
                output.mul_((normalizer - combined).exp().unsqueeze(-1))
                output.add_(
                    partial.float() * (lse.float() - combined).exp().unsqueeze(-1)
                )
                normalizer = combined
            # Do not keep a previous block alive during the next projection.
            del rows, expanded, key, partial, lse
        return output.to(query.dtype)

    def _forward_prefill(self, positions, hidden, ctx, comm_manager):
        backend = ascend_mla_prefill_backend(ctx.attn_backend)
        plan = backend._prefill_plan
        if plan is None or hidden.shape[0] != plan.row_counts[backend._dcp.rank]:
            raise ValueError("MLA Prefill hidden rows must follow the TP partition")
        if (
            self.use_output_gate
            or self.rotary_emb is not None
            or self.q_lora_rank is None
        ):
            raise ValueError("MLA Prefill requires ungated NoPE Q-LoRA")
        if hidden.shape[0]:
            down = self.fused_qkv_a_proj_with_mqa(hidden, None, torch.bfloat16)
            qa, latent = down.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            qnorm = torch.empty_like(qa)
            self.fused_qk_layernorm(qa, latent[:, : self.kv_lora_rank], qnorm)
            if self.kv_lora_scale is not None:
                latent[:, : self.kv_lora_rank].mul_(self.kv_lora_scale)
            latent = latent.contiguous()
            if ctx.token_to_kv_pool.latent_write_sanitizes:
                latent = torch.nan_to_num(latent)
        else:
            qnorm = hidden.new_empty((0, self.q_lora_rank))
            latent = hidden.new_empty((0, self.kv_lora_rank + self.qk_rope_head_dim))
        # Only the normalized Q latent uses the ordinary TP gather. Current
        # KV gathers directly into its final receive plane on the AICPU stream.
        backend.start_prefill_kv(self.layer_id, latent)
        qnorm = comm_manager.pre_attn_comm(qnorm, ctx)
        query = self.q_b_proj(qnorm)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        if self.q_lora_scale is not None:
            query = query * self.q_lora_scale
        if query.shape[0] != plan.row_end:
            raise ValueError("MLA Prefill Q gather must cover the full token span")
        backend.consume_prefill_kv(
            self.layer_id,
            ctx.attn_backend.padded_write_locations(
                self.attn_mqa, ctx.forward_mode, query.shape[0]
            ),
        )
        attended = query.new_empty(
            (query.shape[0], self.num_local_heads, self.v_head_dim)
        )
        for sequence, (begin, end, prefix) in enumerate(backend._prefill_spans):
            attended[begin:end].copy_(
                self._prefill_sequence(
                    query[begin:end].contiguous(),
                    lambda first, last: backend.read_prefill_rows(
                        sequence, first, last
                    ),
                    prefix=prefix,
                    kv_block_tokens=65536,
                )
            )
        backend.finish_prefill_layer(self.layer_id)
        # This is the ordinary local-head partial. CommManager owns the same
        # output ReduceScatter as before; no complete weight copies are needed.
        return self.o_proj(attended.flatten(1))[0]

    def forward(self, positions, hidden_states, ctx, comm_manager, *, projection_out):
        if (
            ctx.forward_mode.is_extend_or_mixed()
            and self.optimized_prefill
            and ascend_mla_prefill_backend(ctx.attn_backend).native_prefill_enabled
        ):
            if projection_out is not None:
                raise ValueError("MLA Prefill output reduction belongs to CommManager")
            return self._forward_prefill(positions, hidden_states, ctx, comm_manager)
        if (
            self.fused_decode is not None
            and ctx.forward_mode.is_decode()
            and ctx.num_extends == 0
            and positions.numel() in self.fused_decode.outputs
        ):
            if projection_out is not None:
                raise ValueError("Fused MLA uses CommManager for output reduction")
            return self.fused_decode.forward(
                hidden_states, positions, ctx, comm_manager
            )
        if self.use_output_gate:
            return super().forward(
                positions,
                hidden_states,
                ctx,
                comm_manager,
                projection_out=projection_out,
            )
        # Zero local rows do not mean an idle attention-TP group. The inherited
        # projection already handles empty local GEMMs and gathers peer rows.
        q, latent_cache = self._project_q_latent(hidden_states, ctx, comm_manager, None)
        expanded = self._prefill_prologue_before_break(positions, q, latent_cache, ctx)
        output = self._attn(positions, q, latent_cache, ctx, expanded=expanded)
        assert projection_out is None, "Lite uses CommManager for output reduction"
        return self.o_proj(output)[0]


class FusedMLADecode:
    """Fused projections, FP32 packed FIA and the existing direct merge."""

    def __init__(self, layer, *, owner, group, rank):
        self.layer, self.owner, self.rank = layer, owner, rank
        self.prolog, self.weights, self.attention, self.merge = prepare_mla_prolog(
            layer, group=group
        )
        device = layer.q_b_proj.weight.device
        self.outputs = {
            rows: (
                torch.empty(
                    (rows, layer.num_heads, 576), dtype=torch.bfloat16, device=device
                ),
                torch.empty(
                    (rows, layer.q_lora_rank), dtype=torch.bfloat16, device=device
                ),
                torch.empty((rows, 576), dtype=torch.bfloat16, device=device),
                torch.zeros(rows, dtype=torch.int64, device=device),
            )
            for rows in owner.outputs
        }
        self.attention_outputs = {
            rows: (
                torch.empty(
                    (rows, 1, layer.num_heads * 512), dtype=torch.float32, device=device
                ),
                torch.empty(
                    (rows, layer.num_heads, 1), dtype=torch.float32, device=device
                ),
            )
            for rows in self.outputs
        }
        self.backends = weakref.WeakKeyDictionary()

    def backend(self, root):
        if root not in self.backends:
            pending = [root]
            found = []
            while pending:
                node = pending.pop()
                if isinstance(node, AscendMLABackend):
                    found.append(node)
                pending.extend(node.child_backends())
            if len(found) != 1 or tuple(found[0]._dcp.ranks) != self.owner.ranks:
                raise ValueError("Fused MLA needs one matching TP8/DCP8 backend")
            self.backends[root] = found[0]
        return self.backends[root]

    def forward(self, hidden, positions, ctx, comm):
        if self.owner.closed:
            raise RuntimeError("MLA communication runtime is closed")
        layer = self.layer
        backend = self.backend(ctx.attn_backend)
        hidden = comm.pre_attn_comm(hidden, ctx)
        rows = hidden.shape[0]
        query, qnorm, latent, nope_positions = self.outputs[rows]
        locations = ctx.attn_backend.write_locations(layer.attn_mqa, ctx.forward_mode)
        slots = backend.prepare_prolog_metadata(locations)
        cache = ctx.token_to_kv_pool.get_key_buffer(layer.layer_id).view(-1, 576)
        if self.weights["rotary_mode"]:
            positions = positions.to(dtype=torch.int64).contiguous()
        else:
            positions = nope_positions
        stream = torch.npu.current_stream()
        hidden.record_stream(stream)
        slots.record_stream(stream)
        positions.record_stream(stream)
        self.prolog(
            hidden,
            weights=self.weights,
            rank=self.rank,
            window=self.owner.window,
            query=query,
            cache=cache,
            slots=slots,
            qnorm=qnorm,
            latent=latent,
            positions=positions,
        )
        local = backend._current_local
        if local is None:
            raise RuntimeError("MLA decode metadata was not refreshed")
        table, lengths, visible = local
        partial, lse = self.attention(
            query,
            cache=cache.view(-1, backend.kernel_page_size, 1, 576),
            table=table,
            lengths=lengths,
            scale=layer.scaling,
            out=self.attention_outputs[rows][0],
            lse=self.attention_outputs[rows][1],
        )
        producer, maximum, total = self.owner.outputs[rows]
        result = self.merge(
            partial,
            lse=lse,
            visible=visible,
            active=backend._current_active,
            producer=producer,
            maximum=maximum,
            total=total,
            peers=self.owner.tables[rows],
            group=self.owner.group,
        )
        output = torch.empty(
            (rows, layer.num_local_heads * layer.v_head_dim),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        attended = result.transpose(0, 1)
        if layer.head_tp_exchanges(ctx):
            attended = attended.contiguous()
        layer.project_attended_heads(attended, ctx, output, gate=None)
        return layer.o_proj(output)[0]


class AscendMLARuntime:
    """Own checked HCCL views through every referencing MLA decode graph."""

    def __init__(self, layers, mapping):
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg,
        )

        self.layers = tuple(layers)
        self.ranks = tuple(mapping.attn.tp_group)
        if not self.layers or len(self.ranks) != 8 or mapping.attn.kvp_size != 8:
            raise ValueError("Fused MLA requires TP8/DCP8 and nonempty MLA layers")
        self.heads = self.layers[0].num_heads
        if any(layer.num_heads != self.heads for layer in self.layers):
            raise ValueError("Fused MLA resources require uniform head geometry")
        self.closed = False
        rank = self.ranks.index(dist.get_rank())
        group = pg.get_dedicated_device_group(self.ranks, "ascend_mla")
        pg.init_process_group(self.ranks, backend="gloo")
        cpu_group = pg.get_process_group("gloo", self.ranks)
        probe = torch.ones(1, device=self.layers[0].q_b_proj.weight.device)
        dist.all_reduce(probe, group=group)
        self.group = group._get_backend(probe.device).get_hccl_comm_name(
            dist.get_rank()
        )
        self.resources = prepare_mla_resources(
            group=self.group, rank=rank, heads=self.heads, rank_size=len(self.ranks)
        )
        self.window = self.resources.window
        self.outputs = self.resources.outputs
        self.tables = self.resources.tables
        torch.npu.synchronize()
        dist.barrier(group=cpu_group)
        self.runtimes = []
        try:
            for layer in self.layers:
                runtime = FusedMLADecode(
                    layer,
                    owner=self,
                    group=pg.get_device_process_group(self.ranks),
                    rank=rank,
                )
                layer.fused_decode = runtime
                self.runtimes.append(runtime)
        except BaseException:
            self.close()
            raise

    def close(self):
        """Call after draining forwards and destroying captured graphs."""
        if self.closed:
            return
        torch.npu.synchronize()
        for layer in self.layers:
            layer.fused_decode = None
        self.resources.release(self.window)
        self.resources = None
        self.window = None
        self.outputs.clear()
        self.tables.clear()
        self.runtimes.clear()
        self.closed = True
