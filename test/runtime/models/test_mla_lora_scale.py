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

"""``--mla-lora-scale``: folded into the LoRA norm weights, or multiplied at
runtime after q_b_proj / kv_a_layernorm as the trainer does."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

import tokenspeed.runtime.models.longcat_flash as longcat_module
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA
from tokenspeed.runtime.models.longcat_flash import (
    LongcatFlashForCausalLM,
    _lora_norm_scales,
    _RuntimeLongcatDecoderLayer,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

HIDDEN = 64
Q_LORA = 16
KV_LORA = 8
ROPE = 4
HEADS = 2
QK_HEAD = 12
EPS = 1e-6


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()
    return out.to(x.dtype)


def _attention(
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    *,
    q_lora_scale: float | None,
    kv_lora_scale: float | None,
    q_norm_seen: list,
) -> DeepseekV3AttentionMLA:
    attn = object.__new__(DeepseekV3AttentionMLA)
    nn.Module.__init__(attn)
    attn.q_lora_rank = Q_LORA
    attn.kv_lora_rank = KV_LORA
    attn.qk_rope_head_dim = ROPE
    attn.q_lora_scale = q_lora_scale
    attn.kv_lora_scale = kv_lora_scale
    attn.fused_qkv_a_proj_with_mqa = lambda x, block_scale, dtype: (x @ a_weight.T).to(
        dtype
    )

    def fused_qk_layernorm(*, input_q_a, input_kv_a, output_q_a):
        output_q_a.copy_(_rmsnorm(input_q_a, q_norm_weight))
        input_kv_a.copy_(_rmsnorm(input_kv_a, kv_norm_weight))

    attn.fused_qk_layernorm = fused_qk_layernorm

    def q_b_proj(q_norm):
        q_norm_seen.append(q_norm.clone())
        return q_norm @ b_weight.T, None

    attn.q_b_proj = q_b_proj
    return attn


def test_runtime_scale_matches_the_folded_weights_up_to_bf16_rounding():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q_scale = (HIDDEN / Q_LORA) ** 0.5
    kv_scale = (HIDDEN / KV_LORA) ** 0.5
    a_weight = torch.randn(Q_LORA + KV_LORA + ROPE, HIDDEN, dtype=dtype) * 0.2
    b_weight = torch.randn(HEADS * QK_HEAD, Q_LORA, dtype=dtype) * 0.2
    q_norm_weight = torch.rand(Q_LORA, dtype=dtype) + 0.5
    kv_norm_weight = torch.rand(KV_LORA, dtype=dtype) + 0.5
    hidden = torch.randn(5, HIDDEN, dtype=dtype)
    comm = SimpleNamespace(pre_attn_comm=lambda x, ctx: x)

    folded_seen, runtime_seen = [], []
    folded = _attention(
        (q_norm_weight.float() * q_scale).to(dtype),
        (kv_norm_weight.float() * kv_scale).to(dtype),
        a_weight,
        b_weight,
        q_lora_scale=None,
        kv_lora_scale=None,
        q_norm_seen=folded_seen,
    )
    runtime = _attention(
        q_norm_weight,
        kv_norm_weight,
        a_weight,
        b_weight,
        q_lora_scale=q_scale,
        kv_lora_scale=kv_scale,
        q_norm_seen=runtime_seen,
    )
    q_folded, latent_folded = folded._project_q_latent(hidden, None, comm, None)
    q_runtime, latent_runtime = runtime._project_q_latent(hidden, None, comm, None)

    # Same math up to where bf16 rounds: the folded path rounds the scaled
    # weight once, the runtime path rounds the norm output and the product.
    torch.testing.assert_close(q_runtime, q_folded, rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(latent_runtime, latent_folded, rtol=4e-2, atol=4e-2)
    # The rope tail is untouched by either path, and the runtime latent is
    # scaled in place so the cache write sees the scaled value.
    torch.testing.assert_close(latent_runtime[:, KV_LORA:], latent_folded[:, KV_LORA:])
    unscaled_latent = _rmsnorm(
        (hidden @ a_weight.T)[:, Q_LORA : Q_LORA + KV_LORA].to(dtype), kv_norm_weight
    )
    assert not torch.allclose(latent_runtime[:, :KV_LORA], unscaled_latent, rtol=0.2)
    # The DSA indexer reads q_lora before q_b_proj: unscaled under runtime,
    # scaled under folded.
    (folded_q_norm,) = folded_seen
    (runtime_q_norm,) = runtime_seen
    torch.testing.assert_close(
        folded_q_norm.float(), runtime_q_norm.float() * q_scale, rtol=4e-2, atol=4e-2
    )


def test_q_lora_scale_needs_a_q_lora_rank():
    with pytest.raises(ValueError, match="q_lora_scale"):
        DeepseekV3AttentionMLA.__init__(
            object.__new__(DeepseekV3AttentionMLA),
            config=None,
            mapping=SimpleNamespace(attn=SimpleNamespace(tp_size=1)),
            hidden_size=HIDDEN,
            num_heads=HEADS,
            qk_nope_head_dim=QK_HEAD - ROPE,
            qk_rope_head_dim=ROPE,
            v_head_dim=8,
            q_lora_rank=None,
            kv_lora_rank=KV_LORA,
            q_lora_scale=2.0,
        )


def _longcat_config(**overrides):
    fields = dict(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_lora_norm_scales_follow_the_checkpoint_flags():
    assert _lora_norm_scales(_longcat_config()) == (
        (HIDDEN / Q_LORA) ** 0.5,
        (HIDDEN / KV_LORA) ** 0.5,
    )
    assert _lora_norm_scales(_longcat_config(mla_scale_q_lora=False)) == (
        None,
        (HIDDEN / KV_LORA) ** 0.5,
    )
    assert _lora_norm_scales(_longcat_config(q_lora_rank=None)) == (
        None,
        (HIDDEN / KV_LORA) ** 0.5,
    )
    assert _lora_norm_scales(
        _longcat_config(mla_scale_q_lora=False, mla_scale_kv_lora=False)
    ) == (None, None)


def _fake_model_for_post_load():
    attn = SimpleNamespace(
        kv_b_proj=SimpleNamespace(weight=torch.ones(HEADS * (8 + 8), KV_LORA)),
        qk_nope_head_dim=8,
        v_head_dim=8,
        q_a_layernorm=SimpleNamespace(weight=torch.ones(Q_LORA)),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(KV_LORA)),
        w_kc=None,
        w_vc=None,
    )
    # post_load_weights only visits the layers this pipeline stage owns, which
    # it tells apart from PPMissingLayer by type.
    layer = object.__new__(_RuntimeLongcatDecoderLayer)
    layer.self_attn = [attn]
    model = object.__new__(LongcatFlashForCausalLM)
    model.config = _longcat_config()
    model.quant_config = None
    model.model = SimpleNamespace(layers=[layer])
    return model, attn


@pytest.mark.parametrize("mode", ["folded", "runtime"])
def test_post_load_weights_folds_only_under_folded(monkeypatch, mode):
    monkeypatch.setitem(global_server_args_dict, "mla_lora_scale", mode)
    model, attn = _fake_model_for_post_load()
    model.post_load_weights()
    # w_kc / w_vc are split out in both modes.
    assert attn.w_kc.shape == (HEADS, KV_LORA, 8)
    assert attn.w_vc.shape == (HEADS, KV_LORA, 8)
    if mode == "folded":
        expected_q = torch.full((Q_LORA,), (HIDDEN / Q_LORA) ** 0.5)
        expected_kv = torch.full((KV_LORA,), (HIDDEN / KV_LORA) ** 0.5)
    else:
        expected_q = torch.ones(Q_LORA)
        expected_kv = torch.ones(KV_LORA)
    torch.testing.assert_close(attn.q_a_layernorm.weight, expected_q)
    torch.testing.assert_close(attn.kv_a_layernorm.weight, expected_kv)
    # Running it again (a weight-update session reloads) folds once more only
    # under folded, which is why runtime must never touch the weights.
    model.post_load_weights()
    if mode == "runtime":
        torch.testing.assert_close(attn.q_a_layernorm.weight, torch.ones(Q_LORA))


@pytest.mark.parametrize("mode", ["folded", "runtime"])
def test_decoder_layer_hands_the_runtime_scales_to_attention(monkeypatch, mode):
    monkeypatch.setitem(global_server_args_dict, "mla_lora_scale", mode)
    seen: list[dict] = []

    def fake_attention(**kwargs):
        seen.append(kwargs)
        return nn.Module()

    config = _longcat_config(
        num_attention_heads=HEADS,
        qk_nope_head_dim=8,
        qk_rope_head_dim=ROPE,
        v_head_dim=8,
        rms_norm_eps=EPS,
        intermediate_size=32,
        hidden_act="silu",
        rope_scaling=None,
    )
    with (
        mock.patch.object(longcat_module, "_DeepseekV3AttentionMLA", fake_attention),
        mock.patch.object(longcat_module, "_DeepseekV3MLP", lambda **kw: nn.Module()),
        mock.patch.object(
            longcat_module, "_RuntimeLongcatMoE", lambda **kw: nn.Module()
        ),
        mock.patch.object(
            longcat_module, "_get_longcat_moe_quant_config", lambda *a: None
        ),
        mock.patch.object(longcat_module, "_get_rope_theta", lambda config: 10000.0),
        mock.patch.object(longcat_module, "_CommManager", lambda **kw: object()),
    ):
        _RuntimeLongcatDecoderLayer(
            config,
            0,
            mapping=SimpleNamespace(
                has_attn_tp=False, attn=SimpleNamespace(has_qcp=False)
            ),
            prefix="l",
        )
    assert len(seen) == 2
    expected = (
        ((HIDDEN / Q_LORA) ** 0.5, (HIDDEN / KV_LORA) ** 0.5)
        if mode == "runtime"
        else (None, None)
    )
    for kwargs in seen:
        assert (kwargs["q_lora_scale"], kwargs["kv_lora_scale"]) == expected
