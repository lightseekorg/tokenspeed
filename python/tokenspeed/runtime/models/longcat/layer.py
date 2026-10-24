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

"""The LongCat layer program following attention0."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.longcat.cute_dsl import (
    HEADS_TOTAL,
    NUM_TOKENS,
    TOPK,
    LongcatEngine,
    MlpShape,
    MoeShape,
    WorkerShape,
    layer_program,
    make_layer_buffers,
    make_moe_buffers,
    mlp_names,
    plan,
)

from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.models.longcat.attention0 import (
    LongcatAttention0Engine,
    _bf16,
    _check_attention_layer,
    _check_qkv_norms,
)
from tokenspeed.runtime.models.longcat.selection import DSASelection

_PAGE_SIZE = 64
_LOCAL_HEADS = HEADS_TOTAL // 8


def _mlp_operands(mlp, index) -> dict[str, torch.Tensor]:
    """An MLP's direct-TMA weight and scale operands under the layer's names."""
    operands = dict(
        gate_w=mlp.gate_weight,
        gate_ws=mlp.gate_scale,
        gate_ids=torch.zeros((1,), dtype=torch.int32, device=mlp.device),
        gate_alpha=mlp.gate_alpha,
        gate_quant=mlp.gate_quant_scale,
        down_w=mlp.down_weight,
        down_ws=mlp.down_scale,
        down_alpha=mlp.down_alpha,
        post_gamma=mlp.gamma,
    )
    if index == 0:
        operands.update(router_w=mlp.router_weight, router_bias=mlp.router_bias)
    names = mlp_names(index)
    return {names[name]: value for name, value in operands.items()}


class LongcatLayerEngine:
    """Own one layer workspace and share compiled programs with identical geometry."""

    _plan_engines: dict = {}
    _moe_buffers: dict = {}

    def __init__(self, attn, post_attention_layernorm, mapping, mlp0, mlp1, moe):
        self.attn = attn
        self.post_norm = post_attention_layernorm
        self.tp_rank = int(mapping.attn.tp_rank)
        self.device = attn.fused_qkv_a_proj_with_mqa.weight.device
        self._check_layer(attn)
        self.kv_layer_id = int(attn.attn_mqa.layer_id)
        self.mlp0, self.mlp1, self.moe = mlp0, mlp1, moe
        self.collective = None
        self.post_norm_gamma = self.q_gamma = self.kv_gamma = None
        self.post_norm_eps = self.sm_scale = self.k_scale = None
        self.w_qkv_a = self.cos_sin_cache = self.q_b_weight = None
        self.w_kc = self.w_vc_t = self.o_proj_weight = None
        self.moe_ready = torch.zeros((4,), dtype=torch.uint32, device=self.device)[:1]
        self.moe_gen = torch.zeros((4,), dtype=torch.uint32, device=self.device)[:1]
        self._engine = self._workspace = None
        self._ready = False

    @staticmethod
    def _check_layer(attn) -> None:
        width = attn.q_lora_rank + attn.kv_lora_rank + attn.qk_rope_head_dim
        qkv_a = attn.fused_qkv_a_proj_with_mqa.weight
        if (
            qkv_a.dtype != torch.bfloat16
            or qkv_a.shape[0] < width
            or qkv_a.shape[1] != 8192
        ):
            raise ValueError(f"attention1 engine expects bf16 qkv_a [>= {width}, 8192]")
        _check_attention_layer(attn, "attention1")
        _check_qkv_norms(attn, "attention1")
        if getattr(attn, "indexer", None) is not None:
            raise ValueError(
                "attention1 engine serves the selection-reuse branch (no indexer)"
            )
        if getattr(attn.rotary_emb, "is_neox_style", True):
            raise ValueError("attention1 engine implements interleaved RoPE only")

    def _prepare(self) -> None:
        if get_is_capture_mode():
            raise RuntimeError(
                "attention1 engine initialised under CUDA-graph capture; run an "
                "eager warm-up forward first"
            )
        attn = self.attn
        if attn.attn_mqa.fp8_kv_scales is not None:
            raise RuntimeError(
                "attention1 engine needs an unscaled KV cache, got "
                f"{attn.attn_mqa.fp8_kv_scales}"
            )
        shared = LongcatAttention0Engine._collective
        if shared is None:
            raise RuntimeError("LongCat layer transport must be prepared by attention0")
        self.collective = shared
        self.post_norm_gamma = _bf16(self.post_norm.weight.data)
        self.post_norm_eps = float(self.post_norm.variance_epsilon)
        self.sm_scale = float(attn.scaling)
        self.k_scale = 1.0
        self.q_gamma = _bf16(attn.q_a_layernorm.weight.data)
        self.kv_gamma = _bf16(attn.kv_a_layernorm.weight.data)
        width = attn.q_lora_rank + attn.kv_lora_rank + attn.qk_rope_head_dim
        self.w_qkv_a = attn.fused_qkv_a_proj_with_mqa.weight.data[:width]
        rotary = attn.rotary_emb.cos_sin_cache
        self.cos_sin_cache = rotary if rotary.dtype == torch.float32 else rotary.float()
        # Native TP8 shard, only the engine's layouts: w_kc [H, 512(n), 128(k)],
        # w_vc_t [H, 128, 512].
        self.q_b_weight = attn.q_b_proj.weight.data.contiguous()
        self.w_kc = attn.w_kc.transpose(1, 2).contiguous()
        self.w_vc_t = attn.w_vc.transpose(1, 2).contiguous()
        self.o_proj_weight = attn.o_proj.weight.data.contiguous()
        self._ready = True

    def forward(self, hidden_states, residual, ctx, selection: DSASelection, packed):
        kv_cache = ctx.token_to_kv_pool.get_key_buffer(self.kv_layer_id).view(-1, 576)
        if not self._ready:
            if int(ctx.token_to_kv_pool.arena.kv_page_size) != _PAGE_SIZE:
                raise RuntimeError("LongCat layer requires 64-row KV pages")
            self._prepare()
        inputs = LongcatAttention0Engine._step_inputs
        if inputs is None or selection.decode is None:
            raise RuntimeError(
                "LongCat layer requires the current attention0 selection"
            )
        decode = selection.decode
        topk_slots = decode.topk_indices.view(NUM_TOKENS, TOPK)
        topk_lens = decode.topk_lens.view(NUM_TOKENS)
        if topk_slots.dtype != torch.int32 or topk_lens.dtype != torch.int32:
            raise RuntimeError("LongCat layer requires an int32 decode selection")
        (x0, s0), (xm, sm) = packed
        hidden_states = hidden_states.contiguous()
        quant_scale = self.mlp1.input_scale_inv
        step = dict(
            positions=inputs["positions"],
            cache_loc=inputs["out_cache_loc32"],
            kv_cache=kv_cache,
            slots=topk_slots,
            lens=topk_lens,
            quant_global=quant_scale,
            router_x=hidden_states,
            moe_hidden=hidden_states,
            mlp0_residual=residual.contiguous(),
            mlp0_x_fp4=x0,
            mlp0_x_scale=s0.reshape(-1),
            moe_x_fp4=xm,
            moe_x_scale=sm.reshape(-1),
        )
        if self._workspace is None:
            if get_is_capture_mode():
                raise RuntimeError(
                    "LongCat layer requires an eager step before graph capture"
                )
            self._engine, self._workspace = self._build_plan(
                step, kv_cache, quant_scale
            )
        else:
            self._workspace.rebind(**step)
        self._engine(self._workspace)
        operands = self._workspace.operands
        return operands["mlp_norm_out"], operands["mlp_residual_out"]

    def _build_plan(self, step, kv_cache, quant_scale):
        mlp1 = self.mlp1
        shape = WorkerShape(
            heads=_LOCAL_HEADS,
            kv_fp8=kv_cache.dtype == torch.float8_e4m3fn,
            splits=12,
            kv_rows=kv_cache.shape[0],
            cos_rows=self.cos_sin_cache.shape[0],
            cos_fp32=True,
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
            rope_interleaved=True,
            rank=self.tp_rank,
            indexer_rank=None,
            norm_eps=self.post_norm_eps,
            quant_outputs=int(quant_scale is not None),
            oproj_tc=True,
            cluster=2,
        )
        mlp_shape = MlpShape(
            rank=self.tp_rank, eps=float(mlp1.norm.variance_epsilon), index=1
        )
        mlp0_shape = MlpShape(
            rank=self.tp_rank,
            eps=float(self.mlp0.norm.variance_epsilon),
            index=0,
        )
        moe = self.moe
        moe_shape = MoeShape(
            experts=int(moe.gate_alpha.numel()),
            gate_clusters=moe.gate_clusters,
            down_clusters=moe.down_clusters,
            group_splits=moe.down_group_splits,
            identity_scale=1.0 / moe.tp_ep_size,
        )
        # The layer kernel sits between two attention-0 kernels that launch
        # with PDL and trigger their dependents; it does the same, and warms
        # MLP0's and the router's weights in its prologue.
        key = (shape, mlp_shape, mlp0_shape, moe_shape)
        engines = LongcatLayerEngine._plan_engines
        if key not in engines:
            program = layer_program(
                mlp0_shape,
                shape,
                mlp_shape,
                moe_shape,
                pdl=True,
            )
            engines[key] = LongcatEngine(plan(program), use_pdl=True)
        engine = engines[key]
        with torch.no_grad():
            q_b_folded = (self.q_b_weight.float() * self.q_gamma.float()[None, :]).to(
                torch.bfloat16
            )
            operands = dict(
                w_qkv=self.w_qkv_a.contiguous(),
                w_qb=q_b_folded,
            )
        operands.update(make_layer_buffers(shape, self.device))
        operands.update(_mlp_operands(mlp1, 1))
        operands.update(_mlp_operands(self.mlp0, 0))
        # The MoE's scratch and counters are self-generational and the
        # layers' kernels run one after another, so one set per device
        # serves every layer.
        moe_bufs = LongcatLayerEngine._moe_buffers
        if self.device not in moe_bufs:
            moe_bufs[self.device] = make_moe_buffers(self.device)
        operands.update(moe_bufs[self.device])
        weights = moe.weights
        operands.update(
            moe_ready=self.moe_ready,
            local_add_gen=self.moe_gen,
            moe_gate_weight=weights["gate_weight"],
            moe_gate_scale=weights["gate_scale"],
            moe_down_weight=weights["down_weight"],
            moe_down_scale=weights["down_scale"],
            moe_gate_alpha=moe.gate_alpha,
            moe_up_alpha=moe.up_alpha,
            moe_down_alpha=moe.down_alpha,
            moe_out_scale=moe.output_global_scale,
        )
        ws = engine.workspace(
            self.device,
            operands={
                **operands,
                **step,
                "w_kc": self.w_kc,
                "w_vc_t": self.w_vc_t,
                "w_o": self.o_proj_weight,
                "kv_gamma": self.kv_gamma,
                "post_gamma": self.post_norm_gamma,
                "cos_sin": self.cos_sin_cache,
                "bases": self.collective.transport_bases,
                "rotation": self.collective.rotation_flags,
            },
        )
        return engine, ws
