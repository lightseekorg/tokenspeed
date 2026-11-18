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

"""Canonical DSA reloads refresh values, not captured weight addresses."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.attention.dsa import ascend
from tokenspeed.runtime.layers.attention.dsa.ascend import (
    AscendDSAAttention,
    FusedDSADecode,
)


@pytest.fixture
def attention():
    # Supply small canonical checkpoint weights without constructing device kernels.
    layer = AscendDSAAttention.__new__(AscendDSAAttention)
    nn.Module.__init__(layer)
    layer.hidden_size = 8
    layer.kv_lora_rank = 2
    layer.scale_kv_lora = True
    layer.qk_nope_head_dim = layer.v_head_dim = 2
    layer.absorbed_weights_k_major = True
    layer.kv_a_layernorm = nn.LayerNorm(2, bias=False)
    layer.kv_b_proj = nn.Linear(2, 4, bias=False)
    layer.q_b_proj = nn.Linear(2, 2, bias=False)
    layer.o_proj = nn.Linear(2, 2, bias=False)
    layer.mapping = Mapping(rank=0, world_size=1)
    layer.w_kc = layer.w_vc = None
    layer._scales_prepared = False
    layer._prefill_weights = None
    layer.prefill_enabled = True
    layer._fused_decode_runtime = None
    with torch.no_grad():
        layer.kv_a_layernorm.weight.copy_(torch.tensor([0.25, 0.5]))
        layer.process_weights_after_loading()
    return layer


@torch.no_grad()
def test_repeated_startup_does_not_refold_norm(attention):
    attention.process_weights_after_loading()
    assert torch.equal(attention.kv_a_layernorm.weight, torch.tensor([0.5, 1.0]))


@pytest.mark.parametrize("scale_kv_lora", [False, True])
@torch.no_grad()
def test_complete_reloads_fold_each_canonical_norm_once(attention, scale_kv_lora):
    attention.scale_kv_lora = scale_kv_lora
    for raw in (torch.tensor([0.375, 0.625]), torch.tensor([0.25, 0.5])):
        attention.kv_a_layernorm.weight.copy_(raw)
        attention.reload_weights_after_loading()
        attention.process_weights_after_loading()
        assert torch.equal(
            attention.kv_a_layernorm.weight, raw * (2 if scale_kv_lora else 1)
        )


@torch.no_grad()
def test_reload_refreshes_absorbed_and_prefill_weights_in_place(attention):
    captured = (attention.w_kc, attention.w_vc, *attention._prefill_weights)
    pointers = tuple(value.data_ptr() for value in captured)
    for seed in (17, 19):
        torch.manual_seed(seed)
        for parameter in attention.parameters():
            parameter.uniform_(-0.5, 0.5)
        attention.reload_weights_after_loading()
        current = (attention.w_kc, attention.w_vc, *attention._prefill_weights)
        assert pointers == tuple(value.data_ptr() for value in current)
        canonical = attention.kv_b_proj.weight.view(1, 4, 2)
        expected = (
            canonical[:, :2],
            canonical[:, 2:].transpose(1, 2),
            attention.q_b_proj.weight,
            canonical[:, :2],
            canonical[:, 2:].transpose(1, 2),
            attention.o_proj.weight,
        )
        for actual, reference in zip(captured, expected, strict=True):
            assert torch.equal(actual, reference)


def _fused_layer(hidden, heads, index_heads, query_rank, scale_q_lora):
    def weight(shape):
        return SimpleNamespace(weight=torch.randn(shape, dtype=torch.bfloat16) * 0.01)

    index_norm = weight((128,))
    index_norm.variance_epsilon = 1e-5
    return SimpleNamespace(
        hidden_size=hidden,
        num_heads=heads,
        num_local_heads=heads // 8,
        q_lora_rank=query_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_output_gate=False,
        layer_id=0,
        attn_mqa=object(),
        index_init_tokens=16,
        index_local_tokens=1024,
        scale_q_lora=scale_q_lora,
        rotary_emb=None,
        indexer=SimpleNamespace(
            rotary_emb=None,
            index_n_heads=index_heads,
            index_head_dim=128,
            use_rotary=False,
            wq_b=weight((index_heads * 128, query_rank)),
            wk=weight((128, hidden)),
            k_norm=index_norm,
            weights_softmax_scale=0.125,
            weights_proj=SimpleNamespace(weight=torch.randn(index_heads, hidden)),
        ),
        q_a_proj=weight((query_rank, hidden)),
        kv_a_proj_with_mqa=weight((576, hidden)),
        q_b_proj=weight((heads // 8 * 192, query_rank)),
        w_kc=torch.randn(heads // 8, 128, 512, dtype=torch.bfloat16),
        q_a_layernorm=weight((query_rank,)),
        kv_a_layernorm=weight((512,)),
        w_vc=torch.randn(heads // 8, 512, 128, dtype=torch.bfloat16),
        o_proj=weight((hidden, heads // 8 * 128)),
    )


def _operands(runtime):
    return (
        runtime.qb,
        runtime.down,
        runtime.uk,
        runtime.gamma_q,
        runtime.gamma_kv,
        runtime.gamma_index,
        runtime.score_weight,
        runtime.w_vc,
        runtime.w_o,
    )


@pytest.mark.parametrize(
    "geometry", [(2048, 32, 16, 768), (4096, 64, 32, 1536), (4096, 64, 64, 1536)]
)
@pytest.mark.parametrize("scale_q_lora", [False, True])
@torch.no_grad()
def test_fused_reload_matches_fresh_weights_and_preserves_addresses(
    monkeypatch, geometry, scale_q_lora
):
    # Only device allocation, device weight packing and process-group I/O are mocked.
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    def gather(pieces, value, *, group):
        for piece in pieces:
            piece.copy_(value)

    kernels = SimpleNamespace(
        require_fused_prolog=lambda **kwargs: None,
        prepare_projection_weight=lambda value: value.clone(),
    )
    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)
    monkeypatch.setattr(ascend, "ascend_dsa_kernels", lambda: kernels)
    monkeypatch.setattr(ascend.dist, "all_gather", gather)
    layer = _fused_layer(*geometry, scale_q_lora)
    runtime = FusedDSADecode(layer, rank=0, group=object(), eps_q=1e-5, eps_kv=1e-5)
    captured = _operands(runtime)
    addresses = tuple(value.data_ptr() for value in captured)
    original_qb = runtime.qb.clone()
    for value in (
        layer.q_b_proj.weight,
        layer.indexer.wq_b.weight,
        layer.q_a_proj.weight,
        layer.kv_a_proj_with_mqa.weight,
        layer.indexer.wk.weight,
        layer.w_kc,
        layer.indexer.weights_proj.weight,
    ):
        value.add_(0.125)
    runtime.refresh_weights(layer)
    fresh = FusedDSADecode(layer, rank=0, group=object(), eps_q=1e-5, eps_kv=1e-5)
    assert addresses == tuple(value.data_ptr() for value in _operands(runtime))
    assert not torch.equal(runtime.qb, original_qb)
    for actual, expected in zip(captured, _operands(fresh), strict=True):
        assert torch.equal(actual, expected)
    assert runtime.query_scale == fresh.query_scale
    assert runtime.score_scale == fresh.score_scale
