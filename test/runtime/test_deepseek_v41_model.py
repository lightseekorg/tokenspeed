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

"""V4.1 backbone checks. Optional upstream comparison uses DEEPSEEK_V41_REFERENCE_DIR.

All checkpoint-sized construction is on meta. CUDA checks use tiny weights and
real FlatKV arenas. Launch the distributed test with torchrun --nproc_per_node=4
and select free devices with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from pathlib import Path
from test.runtime.test_deepseek_v41_cache import R1, R2, _backend, _extend, _tables
from test.runtime.test_deepseek_v41_engram import _mapping, _Tokenizer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import CapturedRows, ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    V41DecoderView,
    V41PrefillSpan,
    V41RowPlan,
)
from tokenspeed.runtime.layers.linear import LinearBase, MergedColumnParallelLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.utils import MoeBackend
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
from tokenspeed.runtime.models import deepseek_v4 as v4
from tokenspeed.runtime.models import deepseek_v41 as v41
from tokenspeed.runtime.models.deepseek_v41 import (
    DeepseekV41Attention,
    DeepseekV41DecoderLayer,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
    DeepseekV41RotaryEmbedding,
    configure_v41_fp8_linear,
    v41_hc_post,
    v41_hc_pre,
    v41_mxfp8_config,
    v41_quantize_fp8,
)
from tokenspeed.runtime.models.deepseek_v41_engram import (
    is_engram_embed_checkpoint_name,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

_ENGRAM_INDEX = "model.safetensors.index.json"


def _bind_engram_tables(model, weights, tmp_path):
    """Write Engram embed tensors to a sliceable safetensors dir; return the rest."""
    embed = {
        name: tensor
        for name, tensor in weights.items()
        if is_engram_embed_checkpoint_name(name)
    }
    rest = {name: tensor for name, tensor in weights.items() if name not in embed}
    if embed:
        save_file(embed, str(tmp_path / "engram.safetensors"), metadata=None)
        (tmp_path / _ENGRAM_INDEX).write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 0},
                    "weight_map": {name: "engram.safetensors" for name in embed},
                }
            )
        )
        model.bind_checkpoint_dir(str(tmp_path))
    return rest


def _config():
    return SimpleNamespace(
        vocab_size=128,
        max_position_embeddings=131072,
        hidden_size=128,
        num_hidden_layers=40,
        num_attention_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=32,
        o_groups=4,
        o_lora_rank=32,
        rms_norm_eps=1e-20,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling={
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        index_n_heads=4,
        index_head_dim=32,
        index_topk=4,
        candidate_source_layer_id=20,
        candidate_block_size=8,
        candidate_topk_blocks=2,
        hc_mult=4,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[72, 204],
        engram_max_ngram_size=4,
        engram_vocab_size=5,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=9,
        num_hash_layers=0,
        n_group=1,
        topk_group=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        hidden_act="silu",
        swiglu_limit=10.0,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        topk_method="noaux_tc",
        expert_dtype="fp4",
        tie_word_embeddings=False,
    )


def _quant():
    return Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
        }
    )


def _ctx(backend, tokens, mode):
    return ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=1,
        num_extends=1 if mode == ForwardMode.EXTEND else 0,
        input_num_tokens=tokens,
        forward_mode=mode,
    )


def _norm(x, weight, eps):
    x32 = x.float()
    return (
        weight.float() * x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    ).to(x.dtype)


def _initialize(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.dtype == torch.uint8:
                param.fill_(122 if "scale" in name else 0x22)
            elif param.dtype == torch.float8_e4m3fn:
                param.copy_((torch.rand(param.shape, device=param.device) - 0.5) * 2)
            elif "norm.weight" in name or name == "norm.weight":
                param.fill_(1)
            elif name.endswith("attn_sink"):
                param.zero_()
            else:
                param.copy_(torch.randn(param.shape, device=param.device) * 0.03)


class _DenseFFN(nn.Module):
    """An explicit test double for isolating backbone order from expert kernels."""

    def __init__(self, config, mapping, quant_config, layer_index, prefix, aux_stream):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.hidden_size, config.hidden_size, dtype=torch.bfloat16)
        )
        self.shared_experts = None
        self.use_mega_moe = False

    def forward(
        self, x, input_ids, num_global_tokens, max_num_tokens_per_gpu, ctx, comm_manager
    ):
        return F.linear(F.silu(x.float()).to(x.dtype), self.weight)


class _Backend:
    def __init__(self, positions, requests, view=None):
        self.meta = SimpleNamespace(positions=positions, request_indices=requests)
        self.view = (
            V41DecoderView(self.meta, None, (), None, None) if view is None else view
        )
        self.calls = []
        self.global_writes = {}
        self.projections = {}

    def query_metadata(self, mode):
        return self.meta

    def decoder_view(self):
        return self.view

    def rows(self):
        return V41RowPlan(self.meta, self.meta, None)

    def compress(self, owner, content, scores, mode, norm_weight, norm_eps):
        assert content.dtype == scores.dtype == torch.float32
        positions, requests = self.meta.positions, self.meta.request_indices
        self.projections[owner] = (content.clone(), scores.clone())
        cutoff = content.shape[0] // 2 * 2
        pooled = (
            content[:cutoff].unflatten(0, (-1, 2))
            * scores[:cutoff].unflatten(0, (-1, 2)).softmax(1)
        ).sum(1)
        output = torch.zeros_like(content)
        output[1:cutoff:2] = pooled
        row_positions, row_requests = torch.full_like(positions, -1), torch.full_like(
            requests, -1
        )
        row_positions[1:cutoff:2] = positions[:cutoff:2]
        row_requests[1:cutoff:2] = requests[:cutoff:2]
        if norm_weight is not None:
            output = _norm(output.to(torch.bfloat16), norm_weight, norm_eps)
        return output, row_positions, row_requests

    def write_global(self, owner, main, index, positions, requests, mode):
        self.global_writes[owner] = (
            main.clone(),
            index.clone(),
            positions.clone(),
            requests.clone(),
        )

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
        self.calls.append(
            (
                layer_id,
                q.clone(),
                swa.clone(),
                index_q,
                index_weights,
                index_process_group,
            )
        )
        assert softmax_scale == q.shape[-1] ** -0.5
        assert any(
            positions is meta.positions and requests is meta.request_indices
            for meta in (self.meta, self.view.metadata)
            for requests in (request_indices,)
        )
        return (q * 0.25).to(q.dtype)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "rank,tp_size,world_size",
    [(0, 1, 1), (0, 4, 4), (1, 4, 4), (2, 4, 4), (3, 4, 4), (0, 1, 4)],
)
def test_lm_head_checkpoint_and_logits_follow_model_dtype(
    device, fused, dtype, rank, tp_size, world_size
):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = _config()
    adapter = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(adapter)
    adapter.mapping = _mapping(rank, tp_size, world_size)
    adapter.encoder_only = False
    adapter.vision = None
    adapter.model = nn.Module()
    adapter.model.config = config
    with set_default_torch_dtype(dtype), torch.device(device):
        adapter.lm_head = adapter.resolve_lm_head(
            SimpleNamespace(text_config=config), _quant(), ""
        )
    head = adapter.lm_head
    replicated = adapter.mapping.attn.has_dp
    assert isinstance(head, v41.ReplicatedLinear if replicated else v41.ParallelLMHead)
    start = 0 if replicated else rank * (config.vocab_size // tp_size)
    checkpoint = torch.zeros(
        config.vocab_size, config.hidden_size, dtype=torch.bfloat16, device="cpu"
    )
    checkpoint[start : start + 2, 0] = 1
    checkpoint[start + 1, 1] = 2**-9
    adapter.load_weights([("head.weight", checkpoint)])
    local = checkpoint[start : start + head.weight.shape[0]].to(
        device=device, dtype=dtype
    )
    assert head.weight.dtype == dtype
    torch.testing.assert_close(head.weight, local, rtol=0, atol=0)

    hidden = torch.ones(1, config.hidden_size, dtype=torch.bfloat16, device=device)
    expected = F.linear(hidden.to(dtype), local, bias=None)
    # BF16 rounds the near tie; wider dtypes retain the second token's margin.
    expected_token = 0 if dtype == torch.bfloat16 else 1
    assert expected.argmax(-1).item() == expected_token

    processor = adapter.resolve_logits_processor(SimpleNamespace(text_config=config))
    assert not processor._use_fused_lm_head
    # Check each real TP shard without collectives, including the fused helper
    # even if that helper is enabled for V4.1 in the future.
    processor.skip_all_gather = True
    processor._use_fused_lm_head = fused
    output = processor(
        input_ids=torch.zeros(1, dtype=torch.int64, device=device),
        hidden_states=hidden,
        lm_head=head,
        logits_metadata=LogitsMetadata.from_forward_context(
            _ctx(None, 1, ForwardMode.DECODE)
        ),
        aux_hidden_states=None,
    )
    assert output.next_token_logits.dtype == dtype
    torch.testing.assert_close(output.next_token_logits, expected, rtol=0, atol=0)
    assert output.next_token_logits.argmax(-1).item() == expected_token


def test_reference_fp8_floor_and_rounding():
    levels = [0.0, 1e-7, 1e-4, 448.0 / 64, 449.0 / 64, 448.0, 896.0]
    x = torch.tensor(levels, dtype=torch.float32).unsqueeze(-1).expand(-1, 32).clone()
    x[:, ::2].neg_()
    codes, scales = v41_quantize_fp8(x)
    amax = x.abs().amax(-1).clamp_min(1e-4)
    rounded = torch.tensor(
        [2.0 ** math.ceil(math.log2(float(value))) for value in amax * (1.0 / 448.0)],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        scales.view(torch.float8_e8m0fnu).float().flatten(), rounded, rtol=0, atol=0
    )
    expected = (x / rounded[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(codes.view(torch.uint8), expected.view(torch.uint8))
    assert int(scales[0]) == int(scales[1]) == 105
    assert v41_quantize_fp8(x[:0])[0].shape == (0, 32)


def test_scale_expansion_then_tp_and_merged_sharding():
    config = _config()
    quant = v41_mxfp8_config(_quant())
    for rank in range(4):
        mapping = _mapping(rank, 4, 4)
        attn = DeepseekV41Attention(
            config, mapping, 2, quant, "model.layers.2.attn", aux_stream=None
        )
        for linear in (attn.wq_a_wkv, attn.wq_b, attn.wo_b, attn.indexer.wq_b):
            shape = (linear.output_size // 32, linear.input_size // 32)
            scales = (torch.arange(math.prod(shape)).reshape(shape) % 7 + 123).to(
                torch.uint8
            )
            linear.weight_scale_inv.weight_loader(
                linear.weight_scale_inv, scales.view(torch.float8_e8m0fnu)
            )
            expected = scales.repeat_interleave(32, dim=0)
            if isinstance(linear, v41.ColumnParallelLinear):
                expected = expected.chunk(linear.tp_size, dim=0)[linear.tp_rank]
            elif isinstance(linear, v41.RowParallelLinear):
                expected = expected.chunk(4, dim=1)[rank]
            assert torch.equal(linear.weight_scale_inv, expected)
        merged = MergedColumnParallelLinear(
            input_size=128,
            output_sizes=[128, 128],
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=torch.bfloat16,
            quant_config=quant,
            prefix="shared.gate_up_proj",
            tp_rank=rank,
            tp_size=4,
            tp_group=mapping.attn.tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        configure_v41_fp8_linear(merged, True)
        for shard in (0, 1):
            scales = (torch.arange(16).reshape(4, 4) + 100 + shard).to(torch.uint8)
            merged.weight_scale_inv.weight_loader(
                merged.weight_scale_inv, scales, shard
            )
            expected = scales.repeat_interleave(32, dim=0).chunk(4, dim=0)[rank]
            assert torch.equal(
                merged.weight_scale_inv[shard * 32 : (shard + 1) * 32], expected
            )
    assert quant.weight_block_size == [1, 32]


def _mix_reference(x, weight, scale, base, eps, hc_eps, iters):
    hc = x.shape[-2]
    flat = x.float().flatten(-2)
    logits = F.linear(flat, weight) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + eps
    )
    pre = (logits[..., :hc] * scale[0] + base[:hc]).sigmoid() + hc_eps
    post = (logits[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]).sigmoid() * 2
    comb = logits[..., 2 * hc :].reshape(*x.shape[:-2], hc, hc) * scale[2] + base[
        2 * hc :
    ].reshape(hc, hc)
    comb = comb.softmax(-1) + hc_eps
    comb /= comb.sum(-2, keepdim=True) + hc_eps
    for _ in range(iters - 1):
        comb /= comb.sum(-1, keepdim=True) + hc_eps
        comb /= comb.sum(-2, keepdim=True) + hc_eps
    return pre, post, comb


def test_single_pass_two_layer_chain_and_comb_orientation(monkeypatch):
    monkeypatch.setattr(v41, "DeepseekV41MoE", _DenseFFN)
    config = _config()
    config.engram_layer_ids = []
    positions = torch.arange(3)
    backend = _Backend(positions, torch.zeros(3, dtype=torch.int64))
    ctx = _ctx(backend, 3, ForwardMode.EXTEND)
    torch.manual_seed(41)
    layers = [
        DeepseekV41DecoderLayer(
            config, _mapping(0, 1, 1), i, None, f"layers.{i}", None, None, False, "gpu"
        )
        for i in (0, 1)
    ]
    for layer in layers:
        _initialize(layer)
        layer.hc_attn_scale.data.fill_(1)
        layer.hc_ffn_scale.data.fill_(1)
    initial = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    pre = torch.zeros(3, 4)
    pre[:, 0] = 1
    actual, expected = initial.clone(), initial.clone()
    actual_pre, expected_pre = pre.clone(), pre.clone()
    for layer in layers:
        actual, actual_pre = layer(
            actual, actual_pre, positions, torch.arange(3), ctx, backend.rows()
        )
        for sublayer, norm, name in (
            (layer.attn, layer.attn_norm, "attn"),
            (layer.ffn, layer.ffn_norm, "ffn"),
        ):
            pre, post, comb = _mix_reference(
                expected,
                getattr(layer, f"hc_{name}_fn"),
                getattr(layer, f"hc_{name}_scale"),
                getattr(layer, f"hc_{name}_base"),
                1e-20,
                1e-6,
                3,
            )
            collapsed = (
                (expected.float() * expected_pre.unsqueeze(-1))
                .sum(-2)
                .to(expected.dtype)
            )
            x = _norm(collapsed, norm.weight, 1e-20)
            x = (
                sublayer(positions, x, ctx, backend.rows())
                if name == "attn"
                else sublayer(x, None, 3, 3, None, None)
            )
            mixed = torch.stack(
                [
                    sum(
                        comb[:, source, target, None] * expected[:, source].float()
                        for source in range(4)
                    )
                    for target in range(4)
                ],
                dim=1,
            )
            expected = (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(
                x.dtype
            )
            expected_pre = pre
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_pre, expected_pre, rtol=0, atol=0)
    assert not torch.equal(actual_pre, torch.ones_like(actual_pre) / 4)
    torch.testing.assert_close(
        v41_hc_pre(actual, actual_pre),
        (actual.float() * actual_pre[..., None]).sum(-2).to(actual.dtype),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("layer_id", [0, 2, 3, 20, 24])
def test_attention_owner_cast_index_and_grouped_output(layer_id):
    torch.manual_seed(100 + layer_id)
    config = _config()
    attn = DeepseekV41Attention(
        config,
        _mapping(0, 1, 1),
        layer_id,
        None,
        f"layers.{layer_id}.attn",
        aux_stream=None,
    )
    _initialize(attn)
    positions = torch.arange(6)
    requests = torch.zeros(6, dtype=torch.int64)
    backend = _Backend(positions, requests)
    ctx = _ctx(backend, 6, ForwardMode.EXTEND)
    x = torch.randn(6, 128, dtype=torch.bfloat16)
    actual = attn(positions, x, ctx, backend.rows())
    qr = _norm(
        F.linear(x, attn.wq_a_wkv.weight[: config.q_lora_rank]),
        attn.q_norm.weight,
        config.rms_norm_eps,
    )
    unrotated_q = F.linear(qr, attn.wq_b.weight).reshape(6, 4, 64)
    expected_q = attn.rotary_emb(unrotated_q, positions, False)
    _, q, swa, iq, iw, group = backend.calls[-1]
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    assert not torch.allclose(q.float().square().mean(-1), torch.ones(6, 4))
    expected_swa = attn.rotary_emb(
        _norm(
            F.linear(x, attn.wq_a_wkv.weight[config.q_lora_rank :]),
            attn.kv_norm.weight,
            config.rms_norm_eps,
        ),
        positions,
        False,
    )
    torch.testing.assert_close(swa, expected_swa, rtol=0, atol=0)
    output = attn.rotary_emb((q * 0.25).to(q.dtype), positions, True).reshape(6, 4, 64)
    grouped = torch.stack(
        [
            F.linear(output[:, g], attn.wo_a.weight.reshape(4, 32, 64)[g])
            for g in range(4)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        actual, F.linear(grouped.flatten(1), attn.wo_b.weight), rtol=0, atol=0
    )
    assert group is None
    if layer_id in config.kv_source_layer_ids:
        main, index, row_pos, row_req = backend.global_writes[layer_id]
        if config.compress_ratios[layer_id] == 2:
            content, score = backend.projections[layer_id]
            pooled = (
                (content.reshape(3, 2, 64) * score.reshape(3, 2, 64).softmax(1))
                .sum(1)
                .to(torch.bfloat16)
            )
            padded = torch.zeros_like(content, dtype=torch.bfloat16)
            padded[1::2] = pooled
            pooled = padded
            assert attn.compressor.wkv_wgate.weight.dtype == torch.bfloat16
            assert content.dtype == score.dtype == torch.float32
        else:
            pooled = F.linear(x, attn.compressor.wkv.weight)
            assert attn.compressor.wkv.weight.dtype == torch.bfloat16
            assert not hasattr(attn.compressor, "wkv_wgate")
        latent = _norm(pooled, attn.compressor.norm.weight, config.rms_norm_eps)
        torch.testing.assert_close(
            main, attn.rotary_emb(latent, row_pos, False), rtol=0, atol=0
        )
        key = _norm(
            F.linear(latent, attn.indexer.wk.weight),
            attn.indexer.k_norm.weight,
            config.rms_norm_eps,
        )
        torch.testing.assert_close(
            index, attn.rotary_emb(key, row_pos, False), rtol=0, atol=0
        )
        live = row_pos >= 0
        assert row_req[live].tolist() == [0] * int(live.sum())
        assert row_pos[live].tolist() == list(
            range(0, 6, config.compress_ratios[layer_id])
        )
        assert (row_req[~live] == -1).all()
        assert (main[~live] == 0).all() and (index[~live] == 0).all()
    else:
        assert not backend.global_writes and attn.compressor is None
    if layer_id in config.index_source_layer_ids:
        assert iq.shape == (6, 4, 32)
        torch.testing.assert_close(
            iw,
            F.linear(x, attn.indexer.weights_proj.weight) * (32**-0.5 * 4**-0.5),
            rtol=0,
            atol=0,
        )
        if layer_id not in config.kv_source_layer_ids:
            assert attn.indexer.wk is None and attn.indexer.k_norm is None
    else:
        assert iq is iw is None and attn.indexer is None


def test_full_40_layer_backbone_engram_and_final_mix(monkeypatch):
    monkeypatch.setattr(v41, "DeepseekV41MoE", _DenseFFN)
    torch.manual_seed(7)
    config = _config()
    model = DeepseekV41Model(config, _mapping(0, 1, 1), None, "model", False, "gpu")
    _initialize(model)
    model.initialize_engram(_Tokenizer())
    ids = torch.tensor([0, 3, 4, 6])
    positions = torch.arange(4)
    previous = torch.tensor([[-1, -1, -1], [0, -1, -1], [3, 0, -1], [4, 3, 0]])
    mask = torch.ones(4, dtype=torch.bool)
    backend = _Backend(positions, torch.zeros(4, dtype=torch.int64))
    ctx = _ctx(backend, 4, ForwardMode.EXTEND)
    captured = []
    handle = model.layers[-1].register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    actual, aux = model(
        ids,
        positions,
        ctx,
        None,
        None,
        engram_previous_tokens=previous,
        engram_token_mask=mask,
        image_mask=None,
    )
    handle.remove()
    assert [call[0] for call in backend.calls] == list(range(40))
    assert sorted(backend.global_writes) == [2, 8, 14, 20]
    assert all(torch.isfinite(actual).flatten())
    h, last_pre = captured[0]
    torch.testing.assert_close(
        actual, _norm(v41_hc_pre(h, last_pre), model.norm.weight, 1e-20), rtol=0, atol=0
    )
    assert not hasattr(model, "hc_head_fn") and aux is None
    assert not hasattr(model, "engram_previous_tokens")
    with pytest.raises(RuntimeError, match="previous-three"):
        model(
            ids,
            positions,
            ctx,
            None,
            None,
            engram_previous_tokens=None,
            engram_token_mask=mask,
            image_mask=None,
        )
    with pytest.raises(RuntimeError, match="already initialized"):
        model.initialize_engram(_Tokenizer())
    adapter = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(adapter)
    forwarded = adapter.prepare_model_kwargs(
        ctx, ids, {"engram_previous_tokens": previous, "engram_token_mask": mask}
    )
    assert forwarded["engram_previous_tokens"] is previous
    assert forwarded["input_embeds"] is forwarded["pp_inbound"] is None


def test_decoder_narrowing_projects_global_from_all_rows_then_runs_the_tail(
    monkeypatch,
):
    """Layers below the candidate source see every row; the candidate source
    writes its global KV for every row and then narrows to the decoder view;
    every later layer, its collectives and the DSpark taps run on the view;
    the sampled rows are the view's logits rows."""
    monkeypatch.setattr(v41, "DeepseekV41MoE", _DenseFFN)
    torch.manual_seed(7)
    config = _config()
    model = DeepseekV41Model(config, _mapping(0, 1, 1), None, "model", False, "gpu")
    _initialize(model)
    model.initialize_engram(_Tokenizer())
    # Taps at, before and after the narrowing layer all reach the drafter in
    # the narrowed layout ctx.captured_rows reports.
    model.dspark_capture_layers = (19, 20, 39)
    # Request 0 continues past this chunk (one decoder row); request 1
    # completes its prompt (both rows).
    ids = torch.tensor([0, 3, 4, 6, 3, 4])
    positions = torch.tensor([0, 1, 2, 3, 0, 1])
    requests = torch.tensor([0, 0, 0, 0, 1, 1])
    keep_rows = torch.tensor([3, 4, 5])
    tail = SimpleNamespace(
        positions=positions[keep_rows], request_indices=requests[keep_rows]
    )
    view = V41DecoderView(
        tail,
        tail,
        (V41PrefillSpan(0, 0, 3, 1, 3), V41PrefillSpan(1, 1, 0, 2, 0)),
        keep_rows,
        torch.tensor([0, 2]),
    )
    backend = _Backend(positions, requests, view)
    ctx = _ctx(backend, 6, ForwardMode.EXTEND)
    ctx.capture_hidden_mode = CaptureHiddenMode.FULL
    seen = {}

    def observe(layer_id):
        def hook(module, args):
            hidden, _, layer_positions, image_mask, layer_ctx, rows = args
            seen[layer_id] = (
                hidden.shape[0],
                layer_positions.tolist(),
                layer_ctx.collective_num_tokens,
                rows.keep_rows is not None,
            )

        return hook

    handles = [
        model.layers[i].register_forward_pre_hook(observe(i)) for i in (19, 20, 21, 39)
    ]
    previous = torch.tensor([[-1, -1, -1], [0, -1, -1], [3, 0, -1], [4, 3, 0]])
    previous = torch.cat((previous, previous[:2]))
    actual, aux = model(
        ids,
        positions,
        ctx,
        None,
        None,
        engram_previous_tokens=previous,
        engram_token_mask=torch.ones(6, dtype=torch.bool),
        image_mask=None,
    )
    for handle in handles:
        handle.remove()
    assert seen[19] == (6, [0, 1, 2, 3, 0, 1], None, False)
    assert seen[20] == (6, [0, 1, 2, 3, 0, 1], 3, True)
    assert seen[21] == (3, [3, 0, 1], 3, False)
    assert seen[39] == (3, [3, 0, 1], 3, False)
    rows_attended = {layer: q.shape[0] for layer, q, *_ in backend.calls}
    assert all(rows_attended[layer] == 6 for layer in range(20))
    assert all(rows_attended[layer] == 3 for layer in range(20, 40))
    # Every owner wrote from all six rows: the ratio-2 owners their pair
    # rows, the candidate source (which then narrows) every token row.
    assert sorted(backend.global_writes) == [2, 8, 14, 20]
    assert all(w[2].numel() == 6 for w in backend.global_writes.values())
    assert backend.global_writes[20][2].tolist() == positions.tolist()
    assert actual.shape == (2, config.hidden_size)
    assert [h.shape for h in aux] == [(3, config.hidden_size)] * 3
    assert ctx.captured_rows == CapturedRows(tail.positions, ((0, 1), (1, 2)))
    assert torch.isfinite(actual).all()


def test_upstream_rope_and_hc_methods():
    root = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
    if root is None:
        pytest.skip("set DEEPSEEK_V41_REFERENCE_DIR for upstream parity")
    path = Path(root) / "inference/model.py"
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "precompute_freqs_cis",
            "apply_rotary_emb",
        ):
            node.decorator_list = []
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "Block":
            selected.extend(
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef)
                and method.name in ("hc_pre", "hc_post")
            )
    namespace = {"torch": torch, "math": math}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    config = _config()
    pos = torch.tensor([0, 1, 100, 70000])
    x = torch.randn(4, 4, 64, dtype=torch.bfloat16)
    for ratio in (0, 1, 2):
        rope = DeepseekV41RotaryEmbedding(config, ratio)
        freqs = namespace["precompute_freqs_cis"](
            32, 70001, 65536 if ratio else 0, 160000 if ratio else 10000, 16, 32, 1
        )[pos]
        for inverse in (False, True):
            expected = x.clone()
            namespace["apply_rotary_emb"](expected[None, ..., -32:], freqs, inverse)
            torch.testing.assert_close(rope(x, pos, inverse), expected, rtol=0, atol=0)
    residual = torch.randn(2, 3, 4, 32, dtype=torch.bfloat16)
    sublayer = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    pre = torch.rand(2, 3, 4)
    comb = torch.rand(2, 3, 4, 4)
    torch.testing.assert_close(
        v41_hc_pre(residual, pre),
        namespace["hc_pre"](None, residual, pre),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        v41_hc_post(sublayer, residual, pre, comb),
        namespace["hc_post"](None, sublayer, residual, pre, comb),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_exact_fp8_linear_and_engram_method(monkeypatch):
    config = _config()
    quant = v41_mxfp8_config(_quant())
    with torch.device("cuda:0"):
        attn = DeepseekV41Attention(
            config, _mapping(0, 1, 1), 0, quant, "layers.0.attn", aux_stream=None
        )
    linear = attn.wq_a_wkv
    weight = torch.randn_like(linear.weight, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    linear.weight.data.copy_(weight)
    scales = torch.full(
        (linear.output_size // 32, 4), 121, dtype=torch.uint8, device="cuda:0"
    )
    linear.weight_scale_inv.weight_loader(linear.weight_scale_inv, scales)
    linear.quant_method.process_weights_after_loading(linear)
    x = torch.randn(5, 128, dtype=torch.bfloat16, device="cuda:0")
    x[0] = 0
    x[1] *= 1e-6
    actual, _ = linear(x, block_scale=None, output_dtype=None)
    if current_platform().is_hopper:
        # Hopper expands the weights to BF16 and leaves the activations alone.
        assert linear.weight.dtype == torch.bfloat16
        assert linear.weight_scale_inv is None
        expected = F.linear(x.float(), weight.float() / 64).to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
    else:
        codes, sf = v41_quantize_fp8(x)
        dequant = (
            codes.float().unflatten(-1, (-1, 32))
            * sf.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        ).flatten(-2)
        expected = F.linear(dequant, weight.float() / 64).to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.002)
        assert linear.weight.dtype == torch.float8_e4m3fn
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = linear(x, block_scale=None, output_dtype=None)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(
        captured, linear(x, block_scale=None, output_dtype=None)[0], rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("execution_mode", ["eager", "decode", "prefill"])
def test_cuda_40_layer_real_flatkv_and_moe(monkeypatch, tmp_path, execution_mode):
    config = _config()
    config.hidden_size = 256
    config.moe_intermediate_size = 256
    config.num_attention_heads = config.o_groups = config.index_n_heads = 2
    config.head_dim, config.index_head_dim = 512, 128
    config.qk_rope_head_dim, config.q_lora_rank = 64, 128
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    torch.manual_seed(41)
    with torch.device("cuda:0"):
        adapter = DeepseekV41ForCausalLM(
            SimpleNamespace(text_config=config),
            _mapping(0, 1, 1),
            _quant(),
            is_multimodal_active=False,
            mm_attention_backend=None,
        )
    adapter.load_weights(
        _bind_engram_tables(adapter, _checkpoint(config), tmp_path).items()
    )
    adapter.initialize_engram(_Tokenizer())
    model = adapter.model
    for module in model.modules():
        if isinstance(module, LinearBase):
            module.quant_method.process_weights_after_loading(module)
        elif isinstance(module, MoELayer):
            module.process_weights_after_loading(module)
    backend = _backend("cuda:0", 2)
    backend.cache_pool.arena.buffer.zero_()
    tables = _tables("cuda:0")
    meta = _extend(backend, tables, [4], [0], [0], [4])
    ctx = _ctx(backend, 4, ForwardMode.EXTEND)
    ids = torch.tensor([0, 3, 4, 6], device="cuda:0")
    previous = torch.tensor(
        [[-1, -1, -1], [0, -1, -1], [3, 0, -1], [4, 3, 0]], device="cuda:0"
    )
    mask = torch.ones(4, dtype=torch.bool, device="cuda:0")
    actual, _ = model(
        ids,
        meta.positions,
        ctx,
        None,
        None,
        engram_previous_tokens=previous,
        engram_token_mask=mask,
        image_mask=None,
    )
    assert actual.shape == (4, 256) and torch.isfinite(actual).all()
    # All 40 layers use the same decode path; the first decode has no ratio-2 row.
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([0], device="cuda:0"),
        torch.tensor([5], device="cuda:0"),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=False,
    )
    meta = backend.query_metadata(ForwardMode.DECODE)
    decoded, _ = model(
        ids[:1],
        meta.positions,
        _ctx(backend, 1, ForwardMode.DECODE),
        None,
        None,
        engram_previous_tokens=torch.tensor([[6, 4, 3]], device="cuda:0"),
        engram_token_mask=mask[:1],
        image_mask=None,
    )
    assert decoded.shape == (1, 256) and torch.isfinite(decoded).all()
    if execution_mode == "eager":
        return
    if execution_mode == "prefill":
        _assert_chunked_prefill_replays_and_narrows(adapter, backend, tables)
        return

    _assert_decode_graph_matches_eager(
        adapter, backend, tables, "cuda:0", ((6, 1), (7, 1), (8, 1), (1, 0), (9, 1))
    )


@torch.inference_mode()
def _assert_chunked_prefill_replays_and_narrows(adapter, backend, tables):
    """A prompt prefilled in two chunks and the same prompt admitted on a
    prefix hit (replaying the cached window) sample the same next token; the
    decoder runs on one row per non-final chunk and on the last window of a
    final one."""
    device = backend.device
    length, hit, window = 200, 128, 128
    torch.manual_seed(3)
    prompt = torch.randint(0, 7, (length,), device=device)
    history = torch.stack(
        [
            torch.cat((torch.full((d,), -1, device=device), prompt[:-d]))
            for d in (1, 2, 3)
        ],
        dim=1,
    )
    arena = backend.cache_pool.arena.buffer
    arena.zero_()
    # Every window the model hands the backend -- full rows and the decoder
    # view alike -- must plan from host spans; a device snapshot would also
    # look back past the replay start.
    canonical_window = backend._window

    def checked_window(positions, requests, mode):
        found = canonical_window(positions, requests, mode)
        assert found is not None, "decoder rows fell back to a device snapshot"
        return found

    backend._window = checked_window

    def run(request, tables, start, count, replay, prompt_len):
        counts = torch.tensor([count], dtype=torch.int32)
        prefix = torch.tensor([start], dtype=torch.int32)
        backend.init_forward_metadata(
            1,
            1,
            torch.tensor([request], device=device),
            (counts + prefix).to(device),
            ForwardMode.EXTEND,
            block_tables=tables,
            extend_seq_lens=counts.to(device),
            extend_seq_lens_cpu=counts,
            extend_prefix_lens=prefix.to(device),
            extend_prefix_lens_cpu=prefix,
            extend_replay_lens_cpu=torch.tensor([replay], dtype=torch.int32),
            extend_prompt_lens_cpu=torch.tensor([prompt_len], dtype=torch.int32),
            extend_with_prefix=start > 0,
        )
        rows = slice(start, start + count)
        ctx = ForwardContext(
            attn_backend=backend,
            token_to_kv_pool=backend.cache_pool,
            bs=1,
            num_extends=1,
            input_num_tokens=count,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            gather_ids=torch.tensor([count - 1], device=device),
        )
        output = adapter(
            ctx=ctx,
            input_ids=prompt[rows],
            positions=backend.query_metadata(ForwardMode.EXTEND).positions,
            engram_previous_tokens=history[rows],
            engram_token_mask=torch.ones(count, dtype=torch.bool, device=device),
            image_mask=None,
        )
        assert output.next_token_logits.shape[0] == 1
        assert torch.isfinite(output.next_token_logits).all()
        return output.next_token_logits.clone(), backend.decoder_view()

    # Request 0: a non-final chunk keeps one decoder row, the final chunk the
    # prompt's last window.
    _, view = run(0, tables, 0, 72, 0, length)
    assert view.metadata.positions.tolist() == [71]
    assert view.logits_rows.tolist() == [0]
    chunked, view = run(0, tables, 72, length - 72, 0, length)
    assert view.keep_rows is None and view.logits_rows is None
    assert view.metadata.positions.tolist() == list(range(72, length))
    # Request 1 hits request 0's global rows [0, hit) and replays the window
    # before the hit into its own SWA/tail pages; the rows above the hit get
    # private global pages.
    hit_pages = {2: tables[R2][0, : hit // 128], 8: tables[R2][0, : hit // 128]}
    hit_pages[14] = hit_pages[2]
    hit_pages[20] = tables[R1][0, : hit // 64]
    hit_tables = {gid: table.clone() for gid, table in tables.items()}
    hit_tables[R2][1, : hit // 128] = hit_pages[2]
    hit_tables[R1][1, : hit // 64] = hit_pages[20]
    pool = backend.cache_pool
    aliased = {
        (owner, name): getattr(pool, name)(owner)[pages].clone()
        for owner, pages in hit_pages.items()
        for name in ("global_kv", "index_k")
    }
    replayed, view = run(
        1, hit_tables, hit - window, length - (hit - window), window, length
    )
    assert view.metadata.positions.tolist() == list(range(length - window, length))
    # The FP8 projections quantize each activation row alone, so a row's result
    # never depends on its batch; Hopper's BF16 GEMM picks its tiling by M and
    # differs between the replay and the chunk by BF16 rounding.
    if current_platform().is_hopper:
        torch.testing.assert_close(replayed, chunked, rtol=2**-6, atol=2**-7)
    else:
        torch.testing.assert_close(replayed, chunked, rtol=0, atol=0)
    # The replayed rows never rewrote the hit's global rows.
    for (owner, name), before in aliased.items():
        torch.testing.assert_close(
            getattr(pool, name)(owner)[hit_pages[owner]], before, rtol=0, atol=0
        )


def _assert_decode_graph_matches_eager(adapter, backend, tables, device, steps):
    ids = torch.zeros(2, dtype=torch.int64, device=device)
    previous = torch.full((2, 3), -1, dtype=torch.int64, device=device)
    mask = torch.zeros(2, dtype=torch.bool, device=device)
    ctx = _ctx(backend, 2, ForwardMode.DECODE)
    arena = backend.cache_pool.arena.buffer
    before_capture = arena.clone()

    def forward():
        return adapter(
            ctx=ctx,
            input_ids=ids,
            positions=backend.query_metadata(ForwardMode.DECODE).positions,
            engram_previous_tokens=previous,
            engram_token_mask=mask,
            image_mask=None,
        ).next_token_logits

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            backend.init_forward_metadata_capture_cuda_graph(
                2,
                torch.zeros(2, device=device, dtype=torch.int64),
                torch.ones(2, device=device, dtype=torch.int32),
                ForwardMode.DECODE,
                block_tables=tables,
            )
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = forward()
    torch.cuda.current_stream().wait_stream(stream)
    arena.copy_(before_capture)
    for step, (length, actual_bs) in enumerate(steps):
        ids.fill_(7 + step)
        previous[0] = torch.tensor([6 + step, 4 + step, 3 + step], device=device)
        mask[0] = bool(actual_bs)
        before = arena.clone()
        for replay in (False, True):
            backend.refresh_decode_metadata(
                2,
                actual_bs,
                torch.tensor([0], device=device),
                torch.tensor([length], device=device),
                forward_mode=ForwardMode.DECODE,
                block_tables=tables,
                num_extends=0,
                for_graph_replay=replay,
            )
            if not replay:
                expected = forward().clone()
                expected_cache = arena.clone()
                arena.copy_(before)
            else:
                graph.replay()
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)
                torch.testing.assert_close(arena, expected_cache, rtol=0, atol=0)
                assert captured.dtype == torch.float32
                assert torch.isfinite(captured).all()


def test_distributed_attention_tp4(monkeypatch, tmp_path):
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("launch with torchrun --nproc_per_node=4")
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    mapping = _mapping(rank, 4, 4)
    pg_manager.init_distributed(
        mapping,
        distributed_init_method="env://",
        backend="nccl",
        timeout=60,
        device_id=device,
    )
    pg_manager.init_process_group(mapping.attn.tp_group, backend="nccl")
    try:
        config = _config()
        with torch.device(device):
            sharded = DeepseekV41Attention(
                config, mapping, 2, None, "attn", aux_stream=None
            )
            full = DeepseekV41Attention(
                config, _mapping(0, 1, 1), 2, None, "attn", aux_stream=None
            )
        torch.manual_seed(41)
        _initialize(full)
        full_params = dict(full.named_parameters())
        for name, param in sharded.named_parameters():
            source = full_params[name]
            loader = getattr(param, "weight_loader", v41.default_weight_loader)
            if name in (
                "wq_a.weight",
                "wkv.weight",
                "compressor.wkv.weight",
                "compressor.wkv_wgate.weight",
                "wq_a_wkv.weight",
                "indexer.wq_b.weight",
                "indexer.weights_proj.weight",
                "indexer.wk.weight",
            ):
                loader(param, source, shard_id=None, begin_size=None)
            else:
                loader(param, source)
        pos = torch.arange(4, device=device)
        backend = _Backend(pos, torch.zeros_like(pos))
        x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
        out = sharded(pos, x, _ctx(backend, 4, ForwardMode.EXTEND))
        assert backend.calls[-1][-1] is None
        assert sharded.indexer.n_local_heads == config.index_n_heads
        expected = full(
            pos, x, _ctx(_Backend(pos, torch.zeros_like(pos)), 4, ForwardMode.EXTEND)
        )
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.003)
        # Exercise the real packed-weight MegaMoE lifecycle, not the CPU loader
        # plan mocks: load -> finalize -> dense preparation -> full text forward.
        monkeypatch.setattr(v4, "get_moe_backend", lambda: MoeBackend.MEGA_MOE)
        monkeypatch.setattr(v41, "get_moe_backend", lambda: MoeBackend.MEGA_MOE)
        monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
        monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 128)
        monkeypatch.setitem(global_server_args_dict, "max_num_seqs", 8)
        config = _loader_config()
        config.hidden_size = 512
        config.moe_intermediate_size = 256
        config.n_routed_experts = 32
        config.head_dim, config.index_head_dim = 512, 128
        config.qk_rope_head_dim, config.q_lora_rank = 64, 128
        with torch.device(device):
            adapter = DeepseekV41ForCausalLM(
                SimpleNamespace(text_config=config),
                mapping,
                _quant(),
                is_multimodal_active=False,
                mm_attention_backend=None,
            )
        adapter.load_weights(
            _bind_engram_tables(adapter, _checkpoint(config), tmp_path).items()
        )
        adapter.initialize_engram(_Tokenizer())
        for module in adapter.modules():
            if isinstance(module, LinearBase):
                module.quant_method.process_weights_after_loading(module)
        assert all(
            layer.ffn.experts._processed_state is not None
            for layer in adapter.model.layers
        )
        backend = _backend(str(device), 2)
        tables = _tables(str(device))
        meta = _extend(backend, tables, [4], [0], [0], [4])
        ids = torch.tensor([0, 3, 4, 6], device=device)
        previous = torch.tensor(
            [[-1, -1, -1], [0, -1, -1], [3, 0, -1], [4, 3, 0]], device=device
        )
        result, _ = adapter.model(
            ids,
            meta.positions,
            _ctx(backend, 4, ForwardMode.EXTEND),
            None,
            None,
            engram_previous_tokens=previous,
            engram_token_mask=torch.ones(4, dtype=torch.bool, device=device),
            image_mask=None,
        )
        assert result.shape == (4, 512) and torch.isfinite(result).all()
        gathered = [torch.empty_like(result) for _ in range(4)]
        torch.distributed.all_gather(gathered, result)
        for other in gathered:
            torch.testing.assert_close(result, other, rtol=0.01, atol=0.01)
        _assert_decode_graph_matches_eager(
            adapter, backend, tables, device, ((5, 1), (6, 1), (7, 1), (1, 0), (8, 1))
        )
    finally:
        torch.distributed.destroy_process_group()


def _loader_config():
    config = _config()
    config.num_hidden_layers = 3
    config.engram_layer_ids = [1]
    config.engram_num_embeddings = [72]
    config.kv_source_layer_ids = config.index_source_layer_ids = [2]
    config.candidate_source_layer_id = 2
    return config


def _mock_loader_hardware(monkeypatch):
    # Keep real V4 MegaMoE allocation and per-expert loading; only hardware
    # plan/finalization are mocked in CPU/meta checkpoint tests.
    monkeypatch.setattr(v4, "get_moe_backend", lambda: MoeBackend.MEGA_MOE)
    monkeypatch.setattr(v41, "get_moe_backend", lambda: MoeBackend.MEGA_MOE)
    monkeypatch.setattr(v4, "dsv4_mega_moe_plan", lambda **kwargs: object())
    monkeypatch.setattr(pg_manager, "get_device_process_group", lambda group: None)
    monkeypatch.setattr(v4.DeepseekV4MegaMoEExperts, "finalize_weights", Mock())
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)


def _loader_model(monkeypatch, config, rank, device):
    _mock_loader_hardware(monkeypatch)
    wrapper = SimpleNamespace(text_config=config)
    with torch.device(device):
        return DeepseekV41ForCausalLM(
            wrapper,
            _mapping(rank, 4, 4),
            _quant(),
            is_multimodal_active=False,
            mm_attention_backend=None,
        )


def _checkpoint(config):
    """Independent raw checkpoint schema, with deliberately distinct shard values."""
    weights = {}

    def dense(name, n, k, fp8):
        values = (torch.arange(n * k).reshape(n, k) % 13 - 6).float() / 8
        weights[name + ".weight"] = values.to(
            torch.float8_e4m3fn if fp8 else torch.bfloat16
        )
        if fp8:
            shape = ((n + 31) // 32, (k + 31) // 32)
            weights[name + ".scale"] = (
                (torch.arange(math.prod(shape)).reshape(shape) % 7 + 121)
                .to(torch.uint8)
                .view(torch.float8_e8m0fnu)
            )

    h, d = config.hidden_size, config.head_dim
    dense("embed", config.vocab_size, h, False)
    dense("head", config.vocab_size, h, False)
    weights["norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        for kind in ("attn", "ffn"):
            weights[f"{p}.{kind}_norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
            for suffix, shape in (
                ("fn", (24, 4 * h)),
                ("base", (24,)),
                ("scale", (3,)),
            ):
                weights[f"{p}.hc_{kind}_{suffix}"] = torch.full(
                    shape, 0.01, dtype=torch.float32
                )
        a = p + ".attn"
        weights[a + ".attn_sink"] = torch.arange(
            config.num_attention_heads, dtype=torch.float32
        )
        for name, n, k in (
            ("wq_a", config.q_lora_rank, h),
            ("wq_b", config.num_attention_heads * d, config.q_lora_rank),
            ("wkv", d, h),
            (
                "wo_a",
                config.o_groups * config.o_lora_rank,
                config.num_attention_heads * d // config.o_groups,
            ),
            ("wo_b", h, config.o_groups * config.o_lora_rank),
        ):
            dense(a + "." + name, n, k, True)
        for name, n in (("q_norm", config.q_lora_rank), ("kv_norm", d)):
            weights[f"{a}.{name}.weight"] = torch.ones(n, dtype=torch.bfloat16)
        if i in config.kv_source_layer_ids:
            dense(a + ".compressor.wkv", d, h, False)
            if config.compress_ratios[i] == 2:
                dense(a + ".compressor.wgate", d, h, False)
            weights[a + ".compressor.norm.weight"] = torch.ones(d, dtype=torch.bfloat16)
            dense(a + ".indexer.wk", config.index_head_dim, d, False)
            weights[a + ".indexer.k_norm.weight"] = torch.ones(
                config.index_head_dim, dtype=torch.bfloat16
            )
        if i in config.index_source_layer_ids:
            dense(
                a + ".indexer.wq_b",
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
                True,
            )
            dense(a + ".indexer.weights_proj", config.index_n_heads, h, False)
        dense(p + ".ffn.gate", config.n_routed_experts, h, False)
        weights[p + ".ffn.gate.bias"] = torch.arange(
            config.n_routed_experts, dtype=torch.float32
        )
        for shard in ("w1", "w2", "w3"):
            n, k = (
                (h, config.moe_intermediate_size)
                if shard == "w2"
                else (config.moe_intermediate_size, h)
            )
            dense(f"{p}.ffn.shared_experts.{shard}", n, k, True)
            for expert in range(config.n_routed_experts):
                e = f"{p}.ffn.experts.{expert}.{shard}"
                weights[e + ".weight"] = torch.full(
                    (n, k // 2), 128 + expert * 3 + int(shard[-1]), dtype=torch.uint8
                ).view(torch.int8)
                weights[e + ".scale"] = torch.full(
                    (n, k // 32), 120 + expert + int(shard[-1]), dtype=torch.uint8
                ).view(torch.float8_e8m0fnu)
        if i in config.engram_layer_ids:
            rows = config.engram_num_embeddings[config.engram_layer_ids.index(i)]
            weights[p + ".engram.embed.weight"] = (
                torch.arange(rows * config.engram_head_dim).reshape(
                    rows, config.engram_head_dim
                )
                % 7
            ).to(torch.float8_e4m3fn)
            weights[p + ".engram.embed.scale"] = torch.full(
                (rows, config.engram_head_dim // 32), 122, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            for name in ("q_weight", "k_weight"):
                weights[f"{p}.engram.{name}"] = torch.ones(4, h, dtype=torch.bfloat16)
            dense(
                p + ".engram.wkv",
                5 * h,
                3 * config.engram_n_heads * config.engram_head_dim,
                True,
            )
    return weights


@pytest.mark.parametrize(
    "active,encoder_only", [(True, False), (False, False), (True, True)]
)
def test_multimodal_checkpoint_load(monkeypatch, tmp_path, active, encoder_only):
    _mock_loader_hardware(monkeypatch)
    text_config = _loader_config()
    config = SimpleNamespace(
        text_config=text_config,
        vision_config=SimpleNamespace(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            patch_size=2,
            downsample_ratio=2,
            rope_theta=10000,
        ),
        encoder_only=encoder_only,
    )
    with set_default_torch_dtype(torch.bfloat16):
        model = v41.DeepseekV41ForCausalLM(
            config=config,
            mapping=_mapping(0, 4, 4),
            quant_config=_quant(),
            is_multimodal_active=active,
            mm_attention_backend="triton_attn",
        )
    weights = {} if encoder_only else _checkpoint(text_config)
    if active:
        for name, param in model.vision.named_parameters():
            raw = name.replace(".attn.qkv_proj.", ".attn.wqkv.").replace(
                ".attn.proj.", ".attn.wo."
            )
            module = model.vision.get_submodule(name.rpartition(".")[0])
            shape = tuple(param.shape)
            if isinstance(module, LinearBase):
                shape = (
                    (module.output_size, module.input_size)
                    if param.ndim == 2
                    else (module.output_size,)
                )
            weights[raw] = torch.full(shape, 0.25, dtype=param.dtype)
    if not encoder_only:
        for i in range(text_config.num_hidden_layers):
            weights[f"layers.{i}.ffn.gate.bias_vl"] = torch.full(
                (text_config.n_routed_experts,), 0.5, dtype=torch.float32
            )
    rest = _bind_engram_tables(model, weights, tmp_path)
    model.load_weights(("model." + name, tensor) for name, tensor in rest.items())
    if active:
        for param in model.vision.parameters():
            torch.testing.assert_close(param, torch.full_like(param, 0.25))
    if encoder_only:
        assert model.model is model.lm_head is model.logits_processor is None
    else:
        model.set_dspark_layers_to_capture([0, 2])
        assert model.model.dspark_capture_layers == (0, 2)
        embed, head = model.get_embed_and_head()
        assert embed is model.get_input_embeddings().weight
        assert head is model.lm_head.weight
        if active:
            assert "bias_vl" not in model.checkpoint_load_report["skipped"]
            for layer in model.model.layers:
                torch.testing.assert_close(
                    layer.ffn.gate.bias_vl,
                    torch.full_like(layer.ffn.gate.bias_vl, 0.5),
                )
        else:
            assert model.vision is None
            assert model.checkpoint_load_report["skipped"]["bias_vl"] == 3


@pytest.mark.parametrize(
    "rank,prefixed,reverse",
    [(0, False, False), (1, True, True), (2, False, True), (3, True, False)],
)
def test_strict_checkpoint_load_tp4_ep4(monkeypatch, rank, prefixed, reverse, tmp_path):
    config = _loader_config()
    with set_default_torch_dtype(torch.bfloat16):
        model = _loader_model(monkeypatch, config, rank, "cpu")
    weights = _checkpoint(config)
    for name in (
        "vision.blocks.0.weight",
        "aligner.weight",
        "image_start",
        "image_end",
        "image_newline",
        "mtp.layers.0.weight",
        "layers.0.ffn.gate.bias_vl",
    ):
        weights[name] = torch.empty(0)
    rest = _bind_engram_tables(model, weights, tmp_path)
    items = list(rest.items())
    if reverse:
        items.reverse()
    model.load_weights(
        (("model." + name if prefixed else name), tensor) for name, tensor in items
    )
    report = model.checkpoint_load_report
    assert report["skipped"] == {
        "vision": 5,
        "draft": 1,
        "bias_vl": 1,
        "remote_expert": 54,
    }
    assert report["loaded"] == len(weights) - sum(report["skipped"].values())
    assert v4.DeepseekV4MegaMoEExperts.finalize_weights.call_count == 3
    assert model.lm_head.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        model.lm_head.weight,
        weights["head.weight"].chunk(4, dim=0)[rank],
        rtol=0,
        atol=0,
    )
    for i, layer in enumerate(model.model.layers):
        a = f"layers.{i}.attn"
        raw = weights[a + ".wo_a.weight"].float()
        scales = (
            weights[a + ".wo_a.scale"]
            .float()
            .repeat_interleave(32, 0)
            .repeat_interleave(32, 1)
        )
        expected = (raw * scales).bfloat16().chunk(4, 0)[rank]
        torch.testing.assert_close(layer.attn.wo_a.weight, expected, rtol=0, atol=0)
        codes = weights[a + ".wq_b.weight"].chunk(4, 0)[rank]
        if current_platform().is_hopper:
            # Loaded straight into BF16; scales fold in after loading.
            assert layer.attn.wq_b.weight.dtype == torch.bfloat16
            torch.testing.assert_close(
                layer.attn.wq_b.weight, codes.to(torch.bfloat16), rtol=0, atol=0
            )
        else:
            assert layer.attn.wq_b.weight.dtype == torch.float8_e4m3fn
            torch.testing.assert_close(
                layer.attn.wq_b.weight.view(torch.uint8),
                codes.view(torch.uint8),
                rtol=0,
                atol=0,
            )
        shared = layer.ffn.shared_experts.gate_up_proj
        for slot, shard in enumerate(("w1", "w3")):
            source = (
                weights[f"layers.{i}.ffn.shared_experts.{shard}.scale"]
                .view(torch.uint8)
                .repeat_interleave(32, 0)
                .chunk(4, 0)[rank]
            )
            torch.testing.assert_close(
                shared.weight_scale_inv.chunk(2, 0)[slot], source, rtol=0, atol=0
            )
            expert = weights[f"layers.{i}.ffn.experts.{rank}.{shard}.weight"].view(
                torch.uint8
            )
            assert layer.ffn.experts.w13_weight.dtype == torch.uint8
            padded = layer.ffn.experts.w13_weight[0].chunk(2, 0)[slot]
            torch.testing.assert_close(
                padded[: config.moe_intermediate_size], expert, rtol=0, atol=0
            )
            assert not padded[config.moe_intermediate_size :].any()
            scales = layer.ffn.experts.w13_weight_scale[0].chunk(2, 0)[slot]
            expected_scales = weights[
                f"layers.{i}.ffn.experts.{rank}.{shard}.scale"
            ].view(torch.uint8)
            torch.testing.assert_close(
                scales[: config.moe_intermediate_size], expected_scales, rtol=0, atol=0
            )
            assert (scales[config.moe_intermediate_size :] == 127).all()
        down = layer.ffn.experts.w2_weight[0]
        torch.testing.assert_close(
            down[:, : config.moe_intermediate_size // 2],
            weights[f"layers.{i}.ffn.experts.{rank}.w2.weight"].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        assert not down[:, config.moe_intermediate_size // 2 :].any()
        scales = layer.ffn.experts.w2_weight_scale[0]
        expected_scales = weights[f"layers.{i}.ffn.experts.{rank}.w2.scale"].view(
            torch.uint8
        )
        torch.testing.assert_close(
            scales[:, : config.moe_intermediate_size // 32],
            expected_scales,
            rtol=0,
            atol=0,
        )
        assert (scales[:, config.moe_intermediate_size // 32 :] == 127).all()
        assert (
            layer.ffn.shared_experts.gate_up_proj.output_size
            == 2 * config.moe_intermediate_size
        )
    embed = model.model.layers[1].engram.embed
    for field in ("weight", "scale"):
        expected = weights[f"layers.1.engram.embed.{field}"][
            embed.row_start : embed.row_end
        ].view(torch.uint8)
        torch.testing.assert_close(
            getattr(embed, field).view(torch.uint8), expected, rtol=0, atol=0
        )
    assert (
        model.model.layers[2].attn.compressor.wkv_wgate.weight.dtype == torch.bfloat16
    )
    torch.testing.assert_close(
        model.model.layers[2].attn.compressor.wkv_wgate.weight,
        torch.cat(
            (
                weights["layers.2.attn.compressor.wkv.weight"],
                weights["layers.2.attn.compressor.wgate.weight"],
            )
        ),
        rtol=0,
        atol=0,
    )
    assert model.model.config is config
    location = model.get_model_config_for_expert_location(model.config)
    assert (location.num_layers, location.num_logical_experts, location.num_groups) == (
        3,
        4,
        1,
    )


@pytest.mark.parametrize(
    "missing",
    [
        "embed.weight",
        "norm.weight",
        "head.weight",
        "layers.0.hc_attn_fn",
        "layers.0.attn.wo_a.scale",
        "layers.0.attn.wq_b.scale",
        "layers.0.ffn.shared_experts.w3.weight",
        "layers.0.ffn.shared_experts.w1.scale",
        "layers.0.ffn.experts.0.w3.weight",
        "layers.0.ffn.experts.0.w2.scale",
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        "layers.1.engram.wkv.scale",
        "layers.2.attn.indexer.wk.weight",
    ],
)
def test_checkpoint_requires_every_local_constituent(monkeypatch, missing, tmp_path):
    config = _loader_config()
    model = _loader_model(monkeypatch, config, 0, "cpu")
    weights = _checkpoint(config)
    del weights[missing]
    rest = _bind_engram_tables(model, weights, tmp_path)
    with pytest.raises(ValueError, match=re.escape(missing)):
        model.load_weights(rest.items())
    v4.DeepseekV4MegaMoEExperts.finalize_weights.assert_not_called()
    assert not hasattr(model, "checkpoint_load_report")


def test_load_weights_rejects_engram_embed_in_iterator(monkeypatch):
    config = _loader_config()
    model = _loader_model(monkeypatch, config, 0, "cpu")
    weights = _checkpoint(config)
    with pytest.raises(ValueError, match="get_slice"):
        model.load_weights(weights.items())


@pytest.mark.parametrize(
    "bad",
    [
        "layers.40.attn.wq_a.weight",
        "layers.0.typo.weight",
        "layers.0.ffn.experts.4.w1.weight",
        "layers.0.ffn.gate.bias_vl_typo",
        "vision_typo.weight",
        "draft_typo.weight",
        "layers.0.attn.wq_a.bias",
        "layers.0.ffn.experts.0.w1.typo",
    ],
)
def test_checkpoint_rejects_unexpected(monkeypatch, bad):
    model = _loader_model(monkeypatch, _loader_config(), 0, "cpu")
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights([(bad, torch.empty(0))])


def test_checkpoint_rejects_duplicate_alias_and_shapes(monkeypatch):
    config = _loader_config()
    model = _loader_model(monkeypatch, config, 0, "cpu")
    with pytest.raises(TypeError, match="Unsupported"):
        model.load_weights([], strict=False)
    weights = _checkpoint(config)
    name = "layers.0.attn.wq_b.weight"
    with pytest.raises(ValueError, match="Duplicate"):
        model.load_weights([(name, weights[name]), ("model." + name, weights[name])])
    with pytest.raises(ValueError, match="expected"):
        model.load_weights([(name, torch.empty(512, 32, dtype=torch.float8_e4m3fn))])
    with pytest.raises(TypeError, match="FP8 E4M3"):
        model.load_weights([(name, weights[name].bfloat16())])
    name = "layers.0.ffn.experts.0.w1.weight"
    with pytest.raises(TypeError, match="packed FP4"):
        model.load_weights([(name, weights[name].bfloat16())])
    name = "layers.0.attn.wo_a.scale"
    with pytest.raises(TypeError, match="E8M0"):
        model.load_weights([(name, weights[name].float())])
    for name in ("norm.weight", "layers.0.ffn.gate.weight", "layers.1.engram.q_weight"):
        with pytest.raises(TypeError, match="bfloat16"):
            model.load_weights([(name, weights[name].float())])
    name = "layers.0.hc_attn_fn"
    with pytest.raises(TypeError, match="float32"):
        model.load_weights([(name, weights[name].bfloat16())])


@pytest.mark.parametrize("with_scale", [False, True])
def test_bf16_wo_a_checkpoint(monkeypatch, with_scale, tmp_path):
    config = _loader_config()
    model = _loader_model(monkeypatch, config, 0, "cpu")
    weights = _checkpoint(config)
    for i in range(3):
        name = f"layers.{i}.attn.wo_a"
        raw = weights[name + ".weight"].float()
        scales = (
            weights[name + ".scale"]
            .float()
            .repeat_interleave(32, 0)
            .repeat_interleave(32, 1)
        )
        weights[name + ".weight"] = (raw * scales).bfloat16()
        if not with_scale:
            del weights[name + ".scale"]
    rest = _bind_engram_tables(model, weights, tmp_path)
    if with_scale:
        with pytest.raises(ValueError, match="BF16 wo_a"):
            model.load_weights(rest.items())
    else:
        model.load_weights(rest.items())
        assert model.model.layers[0].attn.wo_a.weight.dtype == torch.bfloat16


def test_generic_safetensors_engram_load_is_bounded(monkeypatch, tmp_path):
    config = _loader_config()
    from sympy import nextprime

    config.engram_vocab_size = 90000
    prime, rows = 89999, 0
    for _ in range(6):
        prime = int(nextprime(prime))
        rows += prime
    config.engram_num_embeddings = [rows]
    model = _loader_model(monkeypatch, config, 2, "cpu")
    weights = _checkpoint(config)
    rest = _bind_engram_tables(model, weights, tmp_path)

    embed = model.model.layers[1].engram.embed
    copies = []
    original = embed._copy_rows

    def copy_rows(param, rows, local_start):
        assert rows.device.type == "cpu"
        assert rows.shape[0] <= 65536
        assert rows.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu)
        copies.append((local_start, rows.shape[0]))
        original(param, rows, local_start)

    monkeypatch.setattr(embed, "_copy_rows", copy_rows)
    model.load_weights(rest.items())
    local_rows = embed.row_end - embed.row_start
    assert (
        copies
        == [
            (start, min(65536, local_rows - start))
            for start in range(0, local_rows, 65536)
        ]
        * 2
    )
    assert (
        embed.weight.dtype == torch.float8_e4m3fn and embed.scale.dtype == torch.uint8
    )


@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize(
    "vision,with_images", [(False, False), (True, False), (True, True)]
)
def test_routing_matches_reference_bias_and_normalization(topk, vision, with_images):
    logits = torch.tensor([[-100.0, -99.0, -98.0, -97.0], [0.0, 1.0, 2.0, 3.0]])
    bias = torch.tensor([4.0, 3.0, 0.0, 0.0])
    bias_vl = torch.tensor([0.0, 0.0, 5.0, 0.0]) if vision else None
    image_mask = torch.tensor([False, True]) if with_images else None
    moe = v41.DeepseekV41MoE.__new__(v41.DeepseekV41MoE)
    nn.Module.__init__(moe)
    moe.config = SimpleNamespace(num_experts_per_tok=topk, norm_topk_prob=True)
    moe.gate = SimpleNamespace(
        weight=torch.eye(4), e_score_correction_bias=bias, bias_vl=bias_vl
    )
    moe.hash_indices_dtype = torch.int64
    weights, ids, scores = moe._select_experts(logits, image_mask)
    expected_scores = F.softplus(logits).sqrt()
    expected_bias = torch.stack([bias, bias_vl]) if with_images else bias
    expected_ids = (expected_scores + expected_bias).topk(topk, dim=-1).indices
    expected_weights = expected_scores.gather(1, expected_ids)
    if topk > 1:
        expected_weights /= expected_weights.sum(-1, keepdim=True) + 1e-20
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def test_snapshot_full_text_manifest_and_wo_a_dtype(monkeypatch):
    root = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
    if root is None:
        pytest.skip("set DEEPSEEK_V41_REFERENCE_DIR for full snapshot coverage")
    path = Path(root)
    raw_config = json.loads((path / "config.json").read_text())
    config = SimpleNamespace(
        **raw_config["text_config"],
        num_hash_layers=0,
        n_group=1,
        topk_group=1,
        expert_dtype="fp4",
    )
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    union = set()
    for rank in range(4):
        model = _loader_model(monkeypatch, config, rank, "meta")
        union.update(model._checkpoint_targets())
        assert all(
            layer.ffn.experts.intermediate_size == 2560 for layer in model.model.layers
        )
        assert all(
            layer.ffn.shared_experts.down_proj.input_size == 2304
            for layer in model.model.layers
        )
    union.update(f"layers.{i}.attn.wo_a.scale" for i in range(40))
    skipped = {
        name for name in index if model._skip_checkpoint_weight(name) is not None
    }
    assert union == set(index) - skipped
    assert len(index) == 96085
    for filename in {index[f"layers.{i}.attn.wo_a.weight"] for i in range(40)}:
        with safe_open(str(path / filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in union and name.endswith(".attn.wo_a.weight"):
                    tensor = handle.get_slice(name)
                    assert tensor.get_dtype() == "F8_E4M3"
                    assert tensor.get_shape() == [8192, 4096]
