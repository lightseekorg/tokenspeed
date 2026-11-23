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

"""EAGLE-4 draft monokernel seam (``--draft-decode-monokernel``).

With the switch, the EAGLE-4 draft builds its model body with
:class:`Eagle4DraftMonokernelModel` before the regular body class. A chain
draft step (draft steps 1 and later: one token per request, DECODE mode)
of 1 to ``MAX_ROWS`` requests then runs the draft layer and the draft's
final norm as one launch of
``tokenspeed_kernel.ops.model.eagle4.monokernel.eagle4_draft_layer_decode``
(``docs/design/monokernel.md``). Every other draft forward (the first step,
draft-tree lanes, more requests) runs the regular code, and the target
keeps its own path: this switch is the draft's alone.

* After loading, :func:`enable_eagle4_draft_monokernel` packs the layer
  (a tiled copy of its projections, made inside the weight region of the
  memory saver like the weights it copies) and gives the model one
  :class:`MonokernelState`, whose mailbox scratch is allocated there too.
* The launch's output is the final norm's, which the regular path also
  returns as the draft's next hidden state (``norm_output``), so it is both
  the LM head's input and the next step's hidden state.

When the layer cannot be packed, startup fails and names why.
"""

from __future__ import annotations

import logging

import torch
from tokenspeed_kernel.ops.model.eagle4.monokernel import (
    FusedDecodeLayer,
    eagle4_draft_layer_decode,
    prepare_eagle4_draft_layer,
)

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.rotary_embedding import RotaryEmbedding
from tokenspeed.runtime.models.monokernel import NO_KERNEL, MonokernelState
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

# The most requests of a launch: the kernel's projections serve 16 rows.
MAX_ROWS = 16
PAGE_SIZE = 64
# The dimensions the kernel bakes in (Qwen3.8's EAGLE-4 draft).
KERNEL_SHAPE = {
    "hidden_size": 5120,
    "intermediate_size": 16384,
    "num_attention_heads": 40,
    "num_key_value_heads": 40,
    "head_dim": 128,
}
RMS_NORM_EPS = 1e-6
# window_left: a row attends its request's last 4096 keys, its own included.
WINDOW_LEFT = 4095

NOT_TP1 = "not a single-GPU draft (TP1, no attention DP, no PP)"
OTHER_SHAPE = "dimensions other than the kernel's"
OTHER_LAYER = (
    "a layer other than the kernel's (attn_norm, MoE, NoPE, no norm_output, "
    "or another activation, norm eps, window or RoPE)"
)
NOT_BF16 = "quantized or non-BF16 weights"
NOT_BF16_KV = "a non-BF16 KV cache"


def model_reasons(model: torch.nn.Module) -> list[str]:
    """Why the draft body ``model`` (an EAGLE-4 ``Eagle4LlamaModel``) cannot
    have its decode monokernel; empty when it can."""
    config, mapping = model.config, model.mapping
    layer = model.midlayer
    attn = layer.self_attn
    reasons = []
    if not (
        mapping.attn.tp_size == 1
        and mapping.dense.tp_size == 1
        and not mapping.attn.has_dp
        and mapping.pp_size == 1
    ):
        reasons.append(NOT_TP1)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    shape = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": head_dim,
    }
    if shape != KERNEL_SHAPE:
        reasons.append(OTHER_SHAPE)
    rotary = attn.rotary_emb
    norms = (
        layer.input_layernorm,
        layer.hidden_norm,
        layer.post_attention_layernorm,
        model.norm,
    )
    if (
        layer.eh_proj is not None
        or getattr(config, "enable_moe", False)
        or config.hidden_act != "silu"
        or not model.norm_output
        or any(abs(norm.variance_epsilon - RMS_NORM_EPS) >= 1e-12 for norm in norms)
        or attn.attn.sliding_window_size != WINDOW_LEFT
        or attn.attn.sliding_window_sink != 0
        or attn.attn.logit_cap
        # The base rotary class only: scaled variants keep other tables.
        or type(rotary) is not RotaryEmbedding
        or not rotary.is_neox_style
        or rotary.rotary_dim != head_dim
        or rotary.cos_sin_cache.dtype != torch.float32
    ):
        reasons.append(OTHER_LAYER)
    linears = (
        attn.qkv_proj,
        attn.o_proj,
        layer.mlp.gate_up_proj,
        layer.mlp.down_proj,
    )
    if any(
        not isinstance(linear.quant_method, UnquantizedLinearMethod)
        or linear.weight.dtype != torch.bfloat16
        for linear in linears
    ) or any(norm.weight.dtype != torch.bfloat16 for norm in norms):
        reasons.append(NOT_BF16)
    if global_server_args_dict["kv_cache_dtype"] not in ("auto", "bfloat16", "bf16"):
        reasons.append(NOT_BF16_KV)
    return reasons


def layer_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The draft layer's and final norm's weights by the roles of
    :func:`prepare_eagle4_draft_layer`."""
    layer = model.midlayer
    attn, mlp = layer.self_attn, layer.mlp
    return {
        "input_norm": layer.input_layernorm.weight.data,
        "hidden_norm": layer.hidden_norm.weight.data,
        "post_norm": layer.post_attention_layernorm.weight.data,
        "final_norm": model.norm.weight.data,
        "qkv": attn.qkv_proj.weight.data,
        "o_proj": attn.o_proj.weight.data,
        "gate_up": mlp.gate_up_proj.weight.data,
        "down": mlp.down_proj.weight.data,
    }


def takes_monokernel(rows: int, ctx: ForwardContext) -> bool:
    """Whether a draft forward of ``rows`` token rows takes the launch: a
    chain draft step (DECODE, one row per request: draft-tree lanes have
    several) of 1 to ``MAX_ROWS`` requests without draft narrowing."""
    return (
        ctx.forward_mode.is_decode()
        and rows == ctx.bs
        and 1 <= rows <= MAX_ROWS
        and ctx.draft_narrowing is None
    )


def unserved_error(name: str, reasons: list[str]) -> str:
    """The startup error of a draft ``name`` whose layer cannot be packed."""
    return (
        f"--draft-decode-monokernel: no monokernel for the {name} draft layer "
        f"({'; '.join(reasons)}). Serve without --draft-decode-monokernel."
    )


class Eagle4DraftMonokernelModel:
    """An EAGLE-4 draft body whose chain draft steps run the draft layer and
    the final norm as one launch. Put it before the regular body class in
    the bases; every other forward runs the regular code."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Set after loading by enable_eagle4_draft_monokernel.
        self.draft_monokernel: FusedDecodeLayer | None = None
        self.monokernel_state: MonokernelState | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # A chain step drafts from the previous step's output (hidden_size
        # wide); the first step's target taps go through fc on the regular
        # path.
        if (
            input_embeds is not None
            or hidden_states is None
            or hidden_states.shape[-1] != self.config.hidden_size
            or not takes_monokernel(input_ids.shape[0], ctx)
        ):
            return super().forward(
                input_ids,
                positions,
                ctx,
                input_embeds=input_embeds,
                hidden_states=hidden_states,
            )
        if self.monokernel_state is None:
            raise RuntimeError(
                "EAGLE-4 draft: a chain draft step before "
                "enable_eagle4_draft_monokernel packed the draft layer"
            )
        normed = self._decode(input_ids, positions, ctx, hidden_states)
        # norm_output: the final norm's output is also the next step's hidden.
        return normed, [normed]

    def _decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        state = self.monokernel_state
        state.start_step()
        backend = ctx.attn_backend
        attn = self.midlayer.self_attn.attn
        view = backend.group_view(attn.group_id, ctx.bs)
        if view.kernel_page_size != PAGE_SIZE:
            raise RuntimeError(
                f"EAGLE-4 draft monokernel: KV page size {view.kernel_page_size}, "
                f"the kernel reads {PAGE_SIZE}"
            )
        k_cache, v_cache = ctx.token_to_kv_pool.get_kv_buffer(attn.layer_id)
        leaf = backend.leaves[attn.group_id]
        rows = input_ids.shape[0]
        # TP1: the embedding needs no all-reduce.
        embeds = self.embed_tokens(input_ids)
        normed = torch.empty_like(embeds)
        state.launched = True
        eagle4_draft_layer_decode(
            self.draft_monokernel,
            embeds=embeds,
            hidden=hidden_states.contiguous(),
            normed=normed,
            positions=positions.to(torch.int32),
            # The chain drafter publishes each request's length, the step's
            # token included, here (advance_draft_forward_metadata).
            seq_len=leaf.decode_seq_lens_buffer[:rows],
            slots=backend.write_locations(attn, ctx.forward_mode)[:rows],
            page_table=view.page_table[:rows],
            k_cache=k_cache.view(-1, attn.tp_k_head_num, attn.head_dim),
            v_cache=v_cache.view(-1, attn.tp_v_head_num, attn.head_dim),
            cos_sin=self.midlayer.self_attn.rotary_emb.cos_sin_cache,
            scratch=state.scratch,
            step=state.step[:1],
            # One launch a step: the step counter changes after each.
            layer_index=0,
            compile_only=False,
        )
        state.finish_step()
        return normed


def enable_eagle4_draft_monokernel(model: torch.nn.Module) -> None:
    """Pack the draft layer of ``model`` (an EAGLE-4 draft body built with
    :class:`Eagle4DraftMonokernelModel`) after loading, and give it its
    :class:`MonokernelState`.

    Called inside weight loading's memory-saver region, so the packed copy
    and the scratch are released and restored with the draft's weights.

    Raises:
        ValueError: The layer cannot be packed (the message names why).
    """
    name = type(model).__name__
    reasons = model_reasons(model)
    fused = None
    if not reasons:
        fused = prepare_eagle4_draft_layer(
            tensors=layer_tensors(model), max_rows=MAX_ROWS
        )
        if fused is None:
            reasons.append(NO_KERNEL)
    if reasons:
        raise ValueError(unserved_error(name, reasons))
    device = model.norm.weight.device
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=device)
    model.draft_monokernel = fused
    model.monokernel_state = MonokernelState(scratch)
    logger.info(
        f"Draft decode monokernel: the {name} draft layer and final norm, "
        f"chain draft steps of 1 to {MAX_ROWS} requests"
    )


__all__ = [
    "MAX_ROWS",
    "Eagle4DraftMonokernelModel",
    "enable_eagle4_draft_monokernel",
    "model_reasons",
    "takes_monokernel",
]
