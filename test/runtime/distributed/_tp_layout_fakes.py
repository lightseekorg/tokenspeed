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

"""CPU fakes shared by the gloo tests of the head-sharded attention layouts.

``test_decode_tp_layouts`` (head TP over attention-DP ranks) and
``test_qcp_head_tp`` (head TP over the query shards) build the same tiny
``DeepseekV3AttentionMLA`` from the same weights and run it over gloo; this
module holds what both need so neither imports the other as a test module.
``conftest.py`` in this directory puts the directory on ``sys.path``, so the
module imports by its bare name under every pytest import mode (the spawned
workers inherit that path).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tokenspeed.runtime.distributed.mapping import Mapping

HIDDEN = 32
NUM_HEADS = 8
QK_NOPE, QK_ROPE, V_DIM = 8, 4, 6
Q_LORA, KV_LORA = 16, 12


def init_gloo(rank: int, rendezvous: str, mapping: Mapping) -> None:
    """One gloo world; every group the layouts use is registered under the
    device-backend key as well, so the NCCL-named backend path runs on CPU."""
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    pg_manager.init_distributed(
        mapping,
        distributed_init_method=rendezvous,
        backend="gloo",
        timeout=60,
    )
    # Same creation order on every rank (init_process_group enumerates every
    # group of a shape, so the order is by kind, not by this rank's members);
    # a size-1 group needs no collective.
    for group in (
        mapping.attn.head_tp_group,
        mapping.dense.tp_group,
        mapping.lm_head.tp_group,
    ):
        if len(group) == 1 or pg_manager.has_process_group("nccl", group):
            continue
        pg_manager.init_process_group(group, backend="gloo")
        pg_manager.register_process_group(
            "nccl", group, pg_manager.get_process_group("gloo", group)
        )
    # NCCL (here gloo) collectives instead of symmetric-memory kernels.
    global_server_args_dict["force_deterministic_rsag"] = True
    global_server_args_dict["mapping"] = mapping


class StubCoreAttention:
    """Stands in for ``PagedAttention`` on CPU.

    The prologue asserts the one-row-count contract and marks the RoPE
    channels with the row's position; core attention maps each (token, head)
    query through that token's own latent ("KV"), so the output of a head
    for a token depends on exactly the inputs the real kernel reads. A
    narrowing draft step attends the live rows only (``ctx.gather_ids``).
    """

    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.calls = 0

    def latent_prologue(
        self, query, q_pe, latent_cache, positions, ctx, *, slots, expanded, key_rows
    ):
        assert key_rows is None
        assert expanded is None
        assert query.shape[0] == q_pe.shape[0] == latent_cache.shape[0]
        assert query.shape[0] == positions.shape[0] == slots.shape[0]
        # q_pe is the query's own RoPE channels (head TP, after the exchange)
        # or a view of the q_b output sharing no element with the query.
        rotated = query.clone()
        rotated[..., KV_LORA:] = q_pe * (positions.to(query.dtype) + 1.0)[:, None, None]
        self.latent = latent_cache[:, :KV_LORA].clone()
        return SimpleNamespace(query=rotated)

    def __call__(self, Q, k=None, v=None, positions=None, ctx=None, **kwargs):
        assert k is None and v is None
        self.calls += 1
        latent = self.latent
        if ctx.gather_ids is not None and Q.shape[0] != latent.shape[0]:
            latent = latent.index_select(0, ctx.gather_ids)
        kv_gain = 1.0 + latent.sum(dim=-1)  # [T]
        out = Q[..., :KV_LORA] * kv_gain[:, None, None]
        out = out + 0.01 * Q[..., KV_LORA:].sum(dim=-1, keepdim=True)
        return out.reshape(Q.shape[0], -1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def attention_weights(input_width: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(3)

    def linear(out_features: int, in_features: int) -> torch.Tensor:
        # Fan-in scaled, so the activations stay O(1) through four GEMMs and
        # fp32 reassociation noise stays far below the tolerance.
        return torch.randn(out_features, in_features, generator=gen) / in_features**0.5

    return {
        "qkv_a": linear(Q_LORA + KV_LORA + QK_ROPE, input_width),
        "q_b": linear(NUM_HEADS * (QK_NOPE + QK_ROPE), Q_LORA),
        "kv_b": linear(NUM_HEADS * (QK_NOPE + V_DIM), KV_LORA),
        "o": linear(HIDDEN, NUM_HEADS * V_DIM),
        "q_norm": torch.rand(Q_LORA, generator=gen) + 0.5,
        "kv_norm": torch.rand(KV_LORA, generator=gen) + 0.5,
    }


def build_attention(mapping: Mapping, weights: dict[str, torch.Tensor], cls=None):
    """A ``DeepseekV3AttentionMLA`` (or ``cls``) over ``weights`` with the
    CUDA-only submodules replaced by CPU stand-ins; ``attn_mqa`` is a
    :class:`StubCoreAttention`."""
    from tokenspeed.runtime.models.deepseek_v3 import (
        DeepseekV3AttentionMLA,
        DeepseekV3FusedQkvAProjWithMqa,
        _prepare_mla_kv_b_proj_weights,
    )

    cls = cls or DeepseekV3AttentionMLA
    attn = cls(
        config=SimpleNamespace(rms_norm_eps=1e-6),
        mapping=mapping,
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_DIM,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=128,
        quant_config=None,
        layer_id=0,
        prefix="layers.0.self_attn",
        reduce_attn_results=False,
    )
    input_width = weights["qkv_a"].shape[1]
    if input_width != HIDDEN:
        # The Eagle3 layer feeds [embeds || hidden_states].
        attn.fused_qkv_a_proj_with_mqa = DeepseekV3FusedQkvAProjWithMqa(
            input_width, Q_LORA + KV_LORA + QK_ROPE, bias=False
        )
    attn.fused_qkv_a_proj_with_mqa.weight.data.copy_(weights["qkv_a"])
    attn.q_b_proj.weight_loader(attn.q_b_proj.weight, weights["q_b"])
    attn.kv_b_proj.weight_loader(attn.kv_b_proj.weight, weights["kv_b"])
    attn.o_proj.weight_loader(attn.o_proj.weight, weights["o"])
    attn.q_a_layernorm.weight.data.copy_(weights["q_norm"])
    attn.kv_a_layernorm.weight.data.copy_(weights["kv_norm"])
    attn.w_kc, attn.w_vc = _prepare_mla_kv_b_proj_weights(attn.kv_b_proj.weight, attn)

    def fused_norm(input_q_a, input_kv_a, output_q_a):
        output_q_a.copy_(rms_norm(input_q_a, weights["q_norm"]))
        input_kv_a.copy_(rms_norm(input_kv_a, weights["kv_norm"]))

    # Replace the CUDA-only submodules with CPU stand-ins (plain attributes).
    del attn.fused_qk_layernorm
    attn.fused_qk_layernorm = fused_norm
    del attn.attn_mqa
    attn.attn_mqa = StubCoreAttention(layer_id=0)
    return attn
