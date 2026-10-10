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

"""Isolated adapter contracts, not native runtime/kernel integration.

Run ``python -m pytest -q -p no:cacheprovider test/runtime/test_dots3_note_model.py``.
The adapter definitions and shared loader are executed unchanged, with explicit
CPU linear/kernel doubles in place of native dependencies. This keeps codec,
loading, masking, metadata and padding regressions testable on development hosts.
``DOTS3_NOTE_CHECKPOINT`` optionally enables read-only config/header checks;
``DOTS3_NOTE_NATIVE_TEST=1`` enables the real model import smoke test (no stubs).
Neither these tests nor that import substitute for model generation/evaluation.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import struct
from dataclasses import dataclass, replace
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "python/tokenspeed/runtime"


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *nodes,
    ]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def _method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        method,
    ]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[method_name]


def _config_class():
    path = RUNTIME / "configs/dots3_note.py"
    spec = importlib.util.spec_from_file_location("_dots3_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Dots3NoteConfig


def _config_values():
    return dict(
        vocab_size=152064,
        hidden_size=5120,
        intermediate_size=13824,
        moe_intermediate_size=1536,
        num_hidden_layers=46,
        layer_types=[
            "full_attention" if i == 0 or i % 4 == 1 else "sliding_attention"
            for i in range(46)
        ],
        num_attention_heads=128,
        num_key_value_heads=128,
        q_lora_rank=1024,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_theta=80000000.0,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=64,
        swa_q_lora_rank=1024,
        swa_kv_lora_rank=1024,
        swa_qk_nope_head_dim=192,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=128,
        swa_rope_theta=50000.0,
        sliding_window_size=513,
        attention_gate_type="headwise",
        swa_attention_gate_type="headwise",
        apply_mla_qkv_lora_rescale=True,
        attention_bias=False,
        attention_dropout=0.0,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        hidden_act="silu",
        rms_norm_eps=1e-5,
        max_position_embeddings=393216,
        tie_word_embeddings=False,
        bos_token_id=151643,
        eos_token_id=151645,
    )


def _small_config():
    values = _config_values()
    values.update(
        hidden_size=64,
        q_lora_rank=32,
        swa_q_lora_rank=32,
        kv_lora_rank=32,
        swa_kv_lora_rank=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    return _config_class()(**values)


class _Fp8Config:
    def __init__(self):
        self.weight_block_size = [128, 128]
        self.is_checkpoint_fp8_serialized = True
        self.scale_fmt = None


class _Linear(nn.Module):
    """CPU double with the existing linear return and checkpoint-shard contracts."""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias,
        quant_config,
        prefix,
        tp_rank=0,
        tp_size=1,
        tp_group=None,
        reduce_results=False,
    ):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.shard_axis = 1 if prefix.endswith("o_proj") else 0
        local_in = input_size // tp_size if self.shard_axis == 1 else input_size
        local_out = output_size // tp_size if self.shard_axis == 0 else output_size
        self.weight = nn.Parameter(
            torch.empty(
                (local_out, local_in),
                dtype=(
                    torch.float8_e4m3fn if quant_config is not None else torch.bfloat16
                ),
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = (
            nn.Parameter(
                torch.ones(
                    ((local_out + 127) // 128, (local_in + 127) // 128),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            if quant_config is not None
            else None
        )
        self.calls = []

    def weight_loader(self, param, weight):
        if self.tp_size > 1:
            weight = weight.chunk(self.tp_size, dim=self.shard_axis)[self.tp_rank]
        assert param.shape == weight.shape
        param.data.copy_(weight)

    def forward(self, x):
        self.calls.append(x.detach().clone())
        assert self.weight.dtype == torch.bfloat16
        return F.linear(x, self.weight), None


class _RMSNorm(nn.Module):
    def __init__(self, width, *, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
            * self.weight.float()
        ).to(x.dtype)


def _rotate(x, positions, base):
    freq = base ** (-torch.arange(0, x.shape[-1], 2).float() / x.shape[-1])
    phase = positions.float()[:, None, None] * freq
    real, imag = x.float()[..., ::2], x.float()[..., 1::2]
    return (
        torch.stack(
            (
                real * phase.cos() - imag * phase.sin(),
                real * phase.sin() + imag * phase.cos(),
            ),
            -1,
        )
        .flatten(-2)
        .to(x.dtype)
    )


class _Rope(nn.Module):
    def __init__(
        self, width, *, rotary_dim, max_position, base, rope_scaling, is_neox_style
    ):
        super().__init__()
        assert width == rotary_dim and not is_neox_style and rope_scaling is None
        self.base = base
        self.inputs = []

    def as_rotary(self, positions):
        return SimpleNamespace(positions=positions, base=self.base)

    def forward(self, positions, q, k):
        self.inputs.append((q.detach().clone(), k.detach().clone()))
        return _rotate(q, positions, self.base), _rotate(k, positions, self.base)


def _hadamard(x, *, scale):
    matrix = torch.ones((1, 1), device=x.device)
    while matrix.shape[0] < x.shape[-1]:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return (x.float() @ matrix * scale).to(x.dtype)


def _mla_reference(
    q, k, v, cu_q, cu_k, max_q, max_k, scale, *, is_causal, window_left, solution
):
    assert solution == "triton"
    assert (
        is_causal
        and cu_q.tolist() == [0, q.shape[0]]
        and cu_k.tolist() == [0, k.shape[0]]
    )
    assert max_q == q.shape[0] and max_k == k.shape[0]
    end = k.shape[0] - q.shape[0] + torch.arange(q.shape[0])
    pos = torch.arange(k.shape[0])
    mask = (pos[None, :] <= end[:, None]) & (pos[None, :] >= end[:, None] - window_left)
    # FP64 keeps the CPU oracle independent of tile-dependent FP32 reduction order.
    logits = torch.einsum("thd,shd->hts", q.double(), k.double()) * scale
    probs = logits.masked_fill(~mask[None], -torch.inf).softmax(-1)
    return torch.einsum("hts,shd->thd", probs, v.double()).to(q.dtype)


def _latent_prologue(
    query, q_pe, latent, *, expanded, rotary, cache, solution, override
):
    """CPU kernel double; the real PagedAttention wrapper supplies its descriptor."""
    assert expanded is None and solution is None and override is None
    assert rotary is not None and cache is not None
    rope_dim = q_pe.shape[-1]
    prepared = query.clone()
    prepared[..., -rope_dim:] = _rotate(q_pe, rotary.positions, rotary.base)
    rows = latent.clone()
    rows[:, -rope_dim:] = _rotate(
        latent[:, None, -rope_dim:], rotary.positions, rotary.base
    ).squeeze(1)
    slots = cache.slots
    if cache.write_mask is not None:
        rows, slots = rows[cache.write_mask], slots[cache.write_mask]
    page_size = cache.kv_cache.shape[1]
    cache.kv_cache[slots // page_size, slots % page_size, 0] = rows
    return SimpleNamespace(query=prepared)


def _resolve_slots(slots, placement):
    assert placement is None
    return slots, None


@pytest.fixture
def adapter():
    namespace = dict(
        torch=torch,
        nn=nn,
        F=F,
        math=math,
        replace=replace,
        break_point=lambda fn: fn,
        LinearBase=_Linear,
        ReplicatedLinear=_Linear,
        ColumnParallelLinear=_Linear,
        RowParallelLinear=_Linear,
        RMSNorm=_RMSNorm,
        get_rope=_Rope,
        Fp8Config=_Fp8Config,
        hadamard_transform=_hadamard,
        mla_prefill=_mla_reference,
        DeepseekV3Model=nn.Module,
        global_server_args_dict={"tp_batch_invariant": "none"},
        mla_prologue=_latent_prologue,
        resolve_cache_slots=_resolve_slots,
        workspace_topk_to_global_slots=lambda *, workspace_indices, kv_workspace_slots: torch.where(
            workspace_indices >= 0,
            kv_workspace_slots[workspace_indices.clamp_min(0)],
            -1,
        ),
        add_prefix=lambda name, prefix: f"{prefix}.{name}" if prefix else name,
    )
    _definitions(
        RUNTIME / "models/deepseek_v3.py", ["DeepseekV3DecoderLayer"], namespace
    )
    _definitions(
        RUNTIME / "models/deepseek_nextn.py", ["DeepseekV3DraftDecoderLayer"], namespace
    )
    _definitions(RUNTIME / "layers/quantization/utils.py", ["block_dequant"], namespace)
    _definitions(
        RUNTIME / "layers/attention/page_table.py",
        ["build_prefill_kv_workspace_slots"],
        namespace,
    )

    modes = dict(IntEnum=IntEnum, auto=auto)
    _definitions(RUNTIME / "execution/forward_batch_info.py", ["ForwardMode"], modes)
    namespace["ForwardMode"] = modes["ForwardMode"]
    _definitions(RUNTIME / "layers/paged_attention.py", ["PagedAttention"], namespace)
    # Execute the real base loader: attention reaching it would be fused or skipped.
    tree = ast.parse((RUNTIME / "models/deepseek_v3.py").read_text())
    base = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV3ForCausalLM"
    )
    load = next(
        n
        for n in base.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_weights"
    )
    base.body = [load]
    base.bases = [
        ast.Attribute(
            value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load()
        )
    ]
    namespace.update(
        get_layer_id=lambda name: (
            int(name.split(".")[2]) if name.startswith("model.layers.") else None
        ),
        build_moe_checkpoint_loader=lambda **kw: SimpleNamespace(
            matches=lambda name: False
        ),
        ExpertCheckpointSchema=lambda **kw: kw,
        default_weight_loader=lambda param, value: param.data.copy_(value),
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        base,
                    ],
                    type_ignores=[],
                )
            ),
            "<shared loader>",
            "exec",
        ),
        namespace,
    )
    tree = ast.parse((RUNTIME / "models/dots3_note.py").read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    tree.body.insert(
        0,
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(tree),
            str(RUNTIME / "models/dots3_note.py"),
            "exec",
        ),
        namespace,
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield SimpleNamespace(**namespace)
    torch.set_num_threads(previous)


def _attention(adapter, layer_id, *, quantized=False, rank=0, tp=1):
    attn = adapter.Dots3NoteAttention(
        _small_config(),
        layer_id,
        SimpleNamespace(attn=SimpleNamespace(tp_rank=rank, tp_size=tp, tp_group=None)),
        is_nextn=False,
        quant_config=_Fp8Config() if quantized else None,
        prefix=f"model.layers.{layer_id}.self_attn",
    )
    if not quantized:
        with torch.no_grad():
            for name, param in attn.named_parameters():
                if "layernorm" not in name and "k_norm" not in name:
                    param.copy_((torch.randn(param.shape) * 0.05).to(param.dtype))
        attn.prepare_weights()
    return attn


def test_config_roundtrip_and_explicit_moe_group():
    config = _config_class()(**_config_values())
    assert config.model_type == "dots3_note"
    assert config.architectures == ["Dot3NoteForCausalLM"]
    assert config.n_group == config.topk_group == 1
    assert config.pad_token_id is None
    assert (
        _config_class()(**(_config_values() | {"pad_token_id": 151659})).pad_token_id
        == 151659
    )
    assert [
        i for i, kind in enumerate(config.layer_types) if kind == "full_attention"
    ] == [0, *range(1, 46, 4)]
    assert config.layer_types.count("sliding_attention") == 33
    assert (
        config.eos_token_id == 151645
    )  # Generation policy belongs outside the model adapter.
    restored = _config_class().from_dict(json.loads(config.to_json_string()))
    assert restored.layer_types == config.layer_types
    assert restored.swa_kv_lora_rank == 1024


@pytest.mark.parametrize(
    "changes",
    [
        {"layer_types": ["full_attention"]},
        {"n_group": 8},
        {"topk_group": 0},
        {"apply_mla_qkv_lora_rescale": False},
        {"attention_gate_type": "elementwise"},
        {"sliding_window_size": 512},
        {"rope_scaling": {"factor": 2}},
        {"tie_word_embeddings": True},
    ],
)
def test_config_rejects_unsupported_behavior(changes):
    with pytest.raises(ValueError):
        _config_class()(**(_config_values() | changes))


def test_index_codec_floor_power_of_two_and_signed_weights(adapter):
    x = torch.tensor([0.0, 1e-5, 1.0, 448.0, 449.0], dtype=torch.bfloat16)[
        :, None
    ].expand(-1, 128)
    q8, scale = adapter.quantize_index_rows(x)
    expected = 2.0 ** torch.ceil(
        torch.log2(torch.maximum(x.float().abs().amax(-1), torch.tensor(1e-4)) / 448)
    )
    torch.testing.assert_close(scale, expected, rtol=0, atol=0)
    assert q8.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert torch.isfinite(q8.float()).all()
    assert scale[2] != x[2].float().amax() / 448
    torch.testing.assert_close(
        q8.float(),
        (x.float() / expected[:, None]).to(torch.float8_e4m3fn).float(),
        rtol=0,
        atol=0,
    )


@torch.no_grad()
def test_indexer_trailing_rope_hadamard_and_folded_scales(adapter):
    torch.manual_seed(11)
    attn = _attention(adapter, 0)
    indexer = attn.indexer
    x = torch.randn(3, 64).bfloat16()
    q_lora = torch.randn(3, 32).bfloat16()
    positions = torch.tensor([0, 7, 211])
    query, key, key_scale, weights = indexer(x, q_lora, positions)
    raw_q = F.linear(q_lora, indexer.wq_b.weight).view(3, 64, 128)
    raw_k = F.linear(x, indexer.wk.weight)
    norm_k = F.layer_norm(
        raw_k.float(),
        (128,),
        indexer.k_norm.weight.float(),
        indexer.k_norm.bias.float(),
        1e-6,
    ).bfloat16()
    torch.testing.assert_close(
        indexer.rotary_emb.inputs[-1][0], raw_q[..., 64:], rtol=0, atol=0
    )
    torch.testing.assert_close(
        indexer.rotary_emb.inputs[-1][1], norm_k[:, None, 64:], rtol=0, atol=0
    )
    raw_q[..., 64:] = _rotate(raw_q[..., 64:], positions, 80000000.0)
    norm_k[..., 64:] = _rotate(norm_k[:, None, 64:], positions, 80000000.0).squeeze(1)
    q8, qs = adapter.quantize_index_rows(_hadamard(raw_q, scale=128**-0.5))
    k8, ks = adapter.quantize_index_rows(_hadamard(norm_k, scale=128**-0.5))
    torch.testing.assert_close(query, q8.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(key.float(), k8.float(), rtol=0, atol=0)
    torch.testing.assert_close(key_scale, ks, rtol=0, atol=0)
    raw_weights = F.linear(x, indexer.weights_proj.weight).float()
    torch.testing.assert_close(
        weights, raw_weights * 64**-0.5 * qs * 128**-0.5, rtol=0, atol=0
    )
    assert (weights < 0).any()
    expected_score = ((query.float() @ key.float().T).relu() * weights[..., None]).sum(
        1
    ) * key_scale
    direct_score = (
        ((q8.float() * qs[..., None]) @ (k8.float() * ks[:, None]).T).relu()
        * (raw_weights * 64**-0.5)[..., None]
    ).sum(1) * 128**-0.5
    torch.testing.assert_close(expected_score, direct_score, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("layer_id, scale", [(0, 192**-0.5), (2, 1 / 16)])
@torch.no_grad()
def test_projection_rescale_rope_and_absorbed_grid(adapter, layer_id, scale):
    torch.manual_seed(13)
    attn = _attention(adapter, layer_id)
    x = torch.randn(4, 64).bfloat16()
    q_lora, q, latent, rope = attn.project(x)
    normalized_q = attn.q_a_layernorm(F.linear(x, attn.q_a_proj.weight))
    expected_q = (normalized_q * math.sqrt(64 / 32)).bfloat16()
    torch.testing.assert_close(q_lora, expected_q, rtol=0, atol=0)
    raw = F.linear(x, attn.kv_a_proj_with_mqa.weight)
    expected_latent = (attn.kv_a_layernorm(raw[:, :32]) * math.sqrt(2)).bfloat16()
    torch.testing.assert_close(latent, expected_latent, rtol=0, atol=0)
    assert attn.scaling == scale and attn.attn_mqa.scaling == scale
    raw_q = F.linear(q_lora, attn.q_b_proj.weight).view_as(q)
    torch.testing.assert_close(
        q,
        raw_q,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rope,
        attn.k_rope_only_layernorm(raw[:, 32:]),
        rtol=0,
        atol=0,
    )
    absorbed = attn.absorb_query(q)
    expected = torch.einsum("thd,hdr->thr", q[..., : attn.qk_nope_head_dim], attn.w_kc)
    torch.testing.assert_close(absorbed[..., :32], expected, rtol=0, atol=0)


@torch.no_grad()
def test_kv_b_dequantizes_before_swa_320_row_head_split(adapter):
    attn = _attention(adapter, 2, quantized=True)
    attn.kv_b_proj.weight.fill_(1)
    attn.kv_b_proj.weight_scale_inv.copy_(torch.arange(1, 6).float()[:, None])
    attn.prepare_weights()
    expected = (
        torch.arange(1, 6).repeat_interleave(128).bfloat16()[:, None].expand(640, 32)
    )
    per_head = expected.view(2, 320, 32)
    torch.testing.assert_close(attn.w_kc, per_head[:, :192], rtol=0, atol=0)
    torch.testing.assert_close(
        attn.w_vc, per_head[:, 192:].transpose(1, 2), rtol=0, atol=0
    )
    assert attn.w_kc[1, 0, 0] == 3  # Head 1 begins halfway through scale block 2.


class _Pool:
    def __init__(self, page_size, width, totals):
        pages_per_req = [(n + page_size - 1) // page_size for n in totals]
        pages = 1 + sum(pages_per_req)
        self.cache = torch.empty_strided(
            (pages, page_size, 1, width),
            (page_size * width + 256, width, width, 1),
            dtype=torch.bfloat16,
        )
        self.cache.fill_(float("nan"))
        self.cache[0].zero_()
        self.table = torch.zeros((len(totals), max(pages_per_req)), dtype=torch.int32)
        physical = torch.arange(pages - 1, 0, -1, dtype=torch.int32)
        offset = 0
        for req, count in enumerate(pages_per_req):
            self.table[req, :count] = physical[offset : offset + count]
            offset += count
        self.index_values = torch.zeros((pages * page_size, 128), dtype=torch.bfloat16)
        self.index_scales = torch.ones(pages * page_size)
        self.writes = []
        self.index_writes = []

    def get_key_buffer(self, layer_id):
        return self.cache

    def kv_write_target(self, layer_id, slots, write_mask):
        self.writes.append((layer_id, slots.clone()))
        return SimpleNamespace(
            kv_cache=self.cache, slots=slots, write_mask=write_mask, sanitize=False
        )

    def get_index_k_buffer(self, layer_id):
        return self

    def slots(self, req, first, last):
        pos = torch.arange(first, last)
        return (
            self.table[req, pos // self.cache.shape[1]].long() * self.cache.shape[1]
            + pos % self.cache.shape[1]
        )

    def rows(self, slots):
        return self.cache[slots // self.cache.shape[1], slots % self.cache.shape[1], 0]

    def set_mla_kv_buffer(self, layer, loc, latent, rope):
        self.writes.append((layer.layer_id, loc.clone()))
        self.cache[loc // self.cache.shape[1], loc % self.cache.shape[1], 0] = (
            torch.cat((latent, rope), -1)
        )

    def set_index_k_buffer(self, layer_id, loc, values, scales):
        self.index_writes.append(
            (layer_id, loc.clone(), values.clone(), scales.clone())
        )
        self.index_values[loc] = values.bfloat16()
        self.index_scales[loc] = scales


@dataclass
class _Context:
    attn_backend: object
    token_to_kv_pool: object
    bs: int
    num_extends: int
    input_num_tokens: int
    forward_mode: object
    draft_narrowing: object = None


@torch.no_grad()
def test_swa_cached_prefill_visible_prefix_request_local_and_tiled(adapter):
    torch.manual_seed(17)
    attn = _attention(adapter, 2)
    prefixes, lengths = [600, 3], [270, 2]
    pool = _Pool(32, 96, [p + n for p, n in zip(prefixes, lengths)])
    for req, (prefix, length) in enumerate(zip(prefixes, lengths)):
        # Expired rows remain NaN, including missing old pages: they must not be read.
        loc = pool.slots(req, max(0, prefix - 512), prefix + length)
        rows = (torch.randn(len(loc), 96) * 0.1).bfloat16()
        pool.set_mla_kv_buffer(attn.attn_mqa, loc, rows[:, :32], rows[:, 32:])
    meta = SimpleNamespace(
        page_table=pool.table,
        extend_prefix_lens_cpu=torch.tensor(prefixes),
        extend_seq_lens_cpu=torch.tensor(lengths),
    )
    leaf = SimpleNamespace(
        chunked_prefill_metadata=meta, step_counter=Mock(), kernel_solution="triton"
    )
    ctx = _Context(None, pool, 2, 2, sum(lengths), adapter.ForwardMode.EXTEND)
    q = (torch.randn(sum(lengths), 2, 256) * 0.1).bfloat16()
    actual = attn._swa_prefill(q, ctx, leaf)
    expected = []
    offset = 0
    for req, (prefix, length) in enumerate(zip(prefixes, lengths)):
        first = max(0, prefix - 512)
        rows = pool.rows(pool.slots(req, first, prefix + length))
        kv = F.linear(rows[:, :32], attn.kv_b_proj.weight).view(-1, 2, 320)
        k = torch.cat((kv[..., :192], rows[:, None, 32:].expand(-1, 2, -1)), -1)
        expected.append(
            _mla_reference(
                q[offset : offset + length],
                k,
                kv[..., 192:],
                torch.tensor([0, length]),
                torch.tensor([0, len(rows)]),
                length,
                len(rows),
                1 / 16,
                is_causal=True,
                window_left=512,
                solution="triton",
            )
        )
        offset += length
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert (
        max(x.shape[0] for x in attn.kv_b_proj.calls) <= 512 + adapter._SWA_QUERY_TILE
    )
    assert len(attn.kv_b_proj.calls) == 3
    leaf.step_counter.record_cache.assert_called_once()


class _Backend:
    def __init__(
        self, attn, pool, mode, prefixes, lengths, decode_lengths, *, q_len_per_req
    ):
        self.attn = attn
        self.pool = pool
        self.mode = mode
        self.num_extends = len(prefixes)
        self.leaf = SimpleNamespace(
            kernel_solution="triton",
            batch_invariant=False,
            slot_order="selection",
            chunked_prefill_metadata=SimpleNamespace(
                page_table=pool.table[: self.num_extends],
                seq_lens=torch.tensor(
                    [p + n for p, n in zip(prefixes, lengths)], dtype=torch.int32
                ),
                extend_prefix_lens_cpu=torch.tensor(prefixes),
                extend_seq_lens_cpu=torch.tensor(lengths),
            ),
            forward_decode_metadata=SimpleNamespace(
                num_extends=self.num_extends,
                q_len_per_req=q_len_per_req,
                page_table=pool.table,
                seq_lens_k=torch.tensor(
                    [p + n for p, n in zip(prefixes, lengths)] + decode_lengths,
                    dtype=torch.int32,
                ),
            ),
            step_counter=Mock(),
        )
        self.prefill_locs = (
            torch.cat(
                [
                    pool.slots(i, p, p + n)
                    for i, (p, n) in enumerate(zip(prefixes, lengths))
                ]
            )
            if prefixes
            else torch.empty(0, dtype=torch.int64)
        )
        self.decode_locs = (
            torch.cat(
                [
                    (
                        pool.slots(self.num_extends + i, n - q_len_per_req, n)
                        if n
                        else torch.zeros(q_len_per_req, dtype=torch.int64)
                    )
                    for i, n in enumerate(decode_lengths)
                ]
            )
            if decode_lengths
            else torch.empty(0, dtype=torch.int64)
        )
        self.selected = []

    def leaf_for(self, layer):
        assert layer is self.attn.attn_mqa
        return self.leaf

    def cache_placement(self, layer):
        assert layer is self.attn.attn_mqa
        return None

    def write_locations(self, layer, mode):
        assert layer is self.attn.attn_mqa
        return self.prefill_locs if mode.is_extend() else self.decode_locs

    def _attention(self, q, slots):
        self.selected.append(slots.clone())
        outputs = []
        for query, selected in zip(q, slots):
            selected = selected[selected >= 0]
            if not len(selected):
                outputs.append(
                    query.new_zeros((query.shape[0], self.attn.kv_lora_rank))
                )
                continue
            kv = self.pool.rows(selected).double()
            scores = (query.double() @ kv.T) * self.attn.scaling
            outputs.append(
                (scores.softmax(-1) @ kv[:, : self.attn.kv_lora_rank]).to(q.dtype)
            )
        return torch.stack(outputs).flatten(1)

    def forward_sparse_prefill(
        self,
        *,
        q,
        layer,
        token_to_kv_pool,
        kv_seq_lens,
        topk_slots,
        topk_lens,
        max_seq_len,
    ):
        assert layer is self.attn.attn_mqa and token_to_kv_pool is self.pool
        self.leaf.step_counter.record_cache()
        return self._attention(q, topk_slots)

    def forward(self, q, k, v, layer, pool, mode, bs, save_kv_cache, *, ctx, **kwargs):
        assert mode.is_decode() and not save_kv_cache
        assert k is None and v is None and ctx.num_extends == 0
        width = self.leaf.forward_decode_metadata.q_len_per_req
        assert ctx.input_num_tokens == q.shape[0] == bs * width
        if self.attn.indexer is not None:
            slots = kwargs["topk_indices"]
        else:
            lengths = self.leaf.forward_decode_metadata.seq_lens_k[self.num_extends :]
            selected = [
                pool.slots(self.num_extends + i, max(0, end - 513), end)
                for i, n in enumerate(lengths)
                for row in range(width)
                for end in [max(0, int(n) - width + 1 + row)]
            ]
            slots = torch.full(
                (bs * width, max(1, max(map(len, selected)))), -1, dtype=torch.int64
            )
            for i, row in enumerate(selected):
                slots[i, : len(row)] = row
        return self._attention(q, slots)


def _score_topk(query, weights, slots, pool, topk):
    dots = query.float() @ pool.index_values[slots].float().T
    scores = (dots.relu() * weights[:, None]).sum(0) * pool.index_scales[slots]
    count = min(topk, len(slots))
    out = torch.full((topk,), -1, dtype=torch.int64)
    out[:count] = scores.topk(count).indices
    return out, count


@pytest.mark.parametrize("layer_id", [0, 2])
@pytest.mark.parametrize(
    "batch_invariant,slot_order", [(False, "selection"), (True, "sorted")]
)
@pytest.mark.parametrize("width", [1, 2, 4])
@torch.no_grad()
def test_mixed_forward_gate_cache_writes_padding_and_decode_width(
    adapter, monkeypatch, layer_id, width, batch_invariant, slot_order
):
    torch.manual_seed(23)
    attn = _attention(adapter, layer_id)
    attn.index_topk = 2
    prefixes, lengths, decode_lengths = [2, 1], [3, 1], [2 + width, 0]
    pool = _Pool(32 if attn.use_swa else 64, 96, [5, 2, *decode_lengths])
    for req, count in enumerate([2, 1, 2]):
        loc = pool.slots(req, 0, count)
        rows = (torch.randn(count, 96) * 0.1).bfloat16()
        pool.set_mla_kv_buffer(attn.attn_mqa, loc, rows[:, :32], rows[:, 32:])
        pool.set_index_k_buffer(
            layer_id,
            loc,
            torch.randn(count, 128).to(torch.float8_e4m3fn),
            torch.full((count,), 0.125),
        )
    pool.writes.clear()
    pool.index_writes.clear()
    backend = _Backend(
        attn,
        pool,
        adapter.ForwardMode.MIXED,
        prefixes,
        lengths,
        decode_lengths,
        q_len_per_req=width,
    )
    backend.leaf.batch_invariant = batch_invariant
    backend.leaf.slot_order = slot_order
    monkeypatch.setattr(
        pool, "set_mla_kv_buffer", Mock(side_effect=AssertionError("legacy writer"))
    )
    num_tokens = sum(lengths) + len(decode_lengths) * width
    ctx = _Context(backend, pool, 4, 2, num_tokens, adapter.ForwardMode.MIXED)
    spans = []

    def prefill_topk(
        q,
        w,
        slots,
        starts,
        ends,
        *,
        topk,
        softmax_scale,
        index_k_cache,
        page_size,
        max_logits_bytes,
        batch_invariant,
        slot_order,
        solution,
    ):
        assert batch_invariant == backend.leaf.batch_invariant
        assert slot_order == backend.leaf.slot_order
        assert solution == backend.leaf.kernel_solution == "triton"
        assert softmax_scale == 1.0 and index_k_cache is pool
        assert starts.tolist() == [0, 0, 0, 5]
        assert ends.tolist() == [3, 4, 5, 7]
        spans.append((starts.clone(), ends.clone()))
        results = []
        for qi, wi, first, last in zip(q, w, starts.tolist(), ends.tolist()):
            selected, count = _score_topk(qi, wi, slots[first:last], pool, topk)
            results.append(torch.where(selected >= 0, selected + first, -1))
        return torch.stack(results).int(), (ends - starts).clamp_max(topk).int()

    def decode_topk(
        q,
        w,
        seq_lens,
        table,
        *,
        page_size,
        topk,
        softmax_scale,
        q_len_per_req,
        index_k_cache,
        batch_invariant,
        slot_order,
        solution,
    ):
        assert batch_invariant == backend.leaf.batch_invariant
        assert slot_order == backend.leaf.slot_order
        assert solution == backend.leaf.kernel_solution == "triton"
        assert softmax_scale == 1.0 and q_len_per_req == width
        assert seq_lens.tolist() == decode_lengths
        torch.testing.assert_close(table, pool.table[2:])
        result, counts = [], []
        assert q.shape[0] == w.shape[0] == len(decode_lengths) * width
        for i, (query, weight) in enumerate(zip(q, w)):
            req, row = divmod(i, width)
            length = max(0, decode_lengths[req] - width + 1 + row)
            slots = pool.slots(req + 2, 0, length)
            selected, count = _score_topk(query, weight, slots, pool, topk)
            result.append(
                torch.where(selected >= 0, slots[selected.clamp_min(0)], -1)
                if length
                else selected
            )
            counts.append(count)
        return torch.stack(result).int(), torch.tensor(counts, dtype=torch.int32)

    monkeypatch.setitem(attn.forward.__globals__, "dsa_prefill_topk", prefill_topk)
    monkeypatch.setitem(attn.forward.__globals__, "dsa_decode_topk", decode_topk)
    # Two collective-padding rows beyond ctx, plus one padded decode request.
    x = torch.randn(num_tokens + 2, 64).bfloat16()
    positions = torch.tensor([2, 3, 4, 1, *range(2, 2 + width), *([0] * (width + 2))])
    comm = SimpleNamespace(pre_attn_comm=lambda x, ctx: x)
    output = attn(positions, x, ctx, comm)
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert torch.count_nonzero(output[num_tokens:]) == 0
    assert [len(loc) for _, loc in pool.writes] == [4, 2 * width]
    assert len(pool.index_writes) == (0 if attn.use_swa else 2)
    assert len(spans) == (0 if attn.use_swa else 1)
    _, q, latent, rope = attn.project(x[:num_tokens])
    q[..., attn.qk_nope_head_dim :] = _rotate(
        q[..., attn.qk_nope_head_dim :], positions[:num_tokens], attn.rotary_emb.base
    )
    rope = _rotate(rope[:, None], positions[:num_tokens], attn.rotary_emb.base).squeeze(
        1
    )
    slots = torch.cat((backend.prefill_locs, backend.decode_locs))
    live = slots != 0
    torch.testing.assert_close(
        pool.rows(slots[live]), torch.cat((latent, rope), -1)[live], rtol=0, atol=0
    )
    if attn.use_swa:
        prefill_values = attn._swa_prefill(q[:4], ctx, backend.leaf)
    else:
        prefill_values = attn.expand_values(
            backend._attention(attn.absorb_query(q[:4]), backend.selected[0])
        )
    decode_slots = backend.selected[1 if not attn.use_swa else 0]
    decode_values = attn.expand_values(
        backend._attention(attn.absorb_query(q[4:]), decode_slots)
    )
    gate = torch.sigmoid(F.linear(x[:num_tokens], attn.g_proj.weight))
    expected_input = (
        torch.cat((prefill_values, decode_values)) * gate[..., None]
    ).flatten(1)
    torch.testing.assert_close(attn.o_proj.calls[0], expected_input, rtol=0, atol=0)
    writes_before = len(pool.writes)
    with pytest.raises(ValueError, match="token counts"):
        attn(positions, x, replace(ctx, input_num_tokens=num_tokens - 1), comm)
    assert len(pool.writes) == writes_before


def _loader_model(adapter, *, quantized):
    class Host(adapter.Dot3NoteForCausalLM):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = _small_config()
            self.mapping = SimpleNamespace(moe=SimpleNamespace(ep_rank=0, ep_size=1))
            self.quant_config = _Fp8Config() if quantized else None
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module() for _ in range(3)])
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = _attention(adapter, i, quantized=quantized)
            self.model.norm = nn.LayerNorm(64, dtype=torch.bfloat16)

    return Host()


@torch.no_grad()
def test_loader_intercepts_all_attention_fp8_and_explicit_mtp_only(adapter):
    model = _loader_model(adapter, quantized=True)
    weights = []
    for name, module in model.named_modules():
        if isinstance(module, _Linear):
            shape = module.weight.shape
            value = torch.ones(shape, dtype=torch.float8_e4m3fn)
            scale = torch.full(((shape[0] + 127) // 128, (shape[1] + 127) // 128), 0.25)
            # Scales before weights exercises the unordered checkpoint stream.
            weights.extend(
                [(f"{name}.weight_scale_inv", scale), (f"{name}.weight", value)]
            )
    for name, param in model.named_parameters():
        if "layernorm" in name or "k_norm" in name or name.startswith("model.norm."):
            weights.append((name, torch.full_like(param, 0.75)))
    weights += [
        ("model.mtp.unused.weight", torch.tensor(1)),
        ("model.layers.46.self_attn.unused.weight", torch.tensor(1)),
    ]
    model.load_weights(iter(weights))
    for layer in model.model.layers:
        attn = layer.self_attn
        assert torch.all(attn.q_a_proj.weight.float() == 1)
        assert torch.all(attn.kv_a_proj_with_mqa.weight == 0.25)
        assert torch.all(attn.g_proj.weight == 0.25)
        assert torch.all(attn.w_kc == 0.25) and torch.all(attn.w_vc == 0.25)
        if attn.indexer is not None:
            assert torch.all(attn.indexer.wk.weight.float() == 1)
            assert torch.all(attn.indexer.weights_proj.weight == 0.25)
            assert torch.all(attn.indexer.k_norm.bias == 0.75)
    assert torch.all(model.model.norm.weight == 0.75)
    with pytest.raises(KeyError):
        model.load_weights([("model.layers.47.unknown.weight", torch.tensor(1))])
    with pytest.raises(KeyError):
        model.load_weights([("model.audio.unknown.weight", torch.tensor(1))])
    with pytest.raises(ValueError, match="Incomplete"):
        model.load_weights(
            [
                (
                    "model.layers.0.self_attn.g_proj.weight",
                    torch.ones((4, 64), dtype=torch.float8_e4m3fn),
                )
            ]
        )


def test_checkpoint_config_and_header_shapes_readonly(adapter):
    """Check the selected checkpoint's assets and FP8 headers without loading weights."""
    directory = os.environ.get("DOTS3_NOTE_CHECKPOINT")
    if directory is None:
        pytest.skip(
            "Set DOTS3_NOTE_CHECKPOINT for read-only checkpoint header validation"
        )
    from transformers import AutoTokenizer

    root = Path(directory)
    source_config = json.loads((root / "config.json").read_text())
    config = _config_class().from_dict(source_config)
    generation = json.loads((root / "generation_config.json").read_text())
    tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    token_ids = {
        token["content"]: int(token_id)
        for token_id, token in tokenizer_config["added_tokens_decoder"].items()
        if token["special"]
    }
    for token in ("<|endoftext|>", "<|endofassistant|>", "<|im_end|>"):
        assert 0 <= token_ids[token] < config.vocab_size
        assert tokenizer.encode(token, add_special_tokens=False) == [token_ids[token]]
    assert config.eos_token_id == source_config["eos_token_id"]
    assert config.eos_token_id in token_ids.values()
    assert tokenizer.eos_token == tokenizer_config["eos_token"] == "<|endoftext|>"
    assert tokenizer.eos_token_id == token_ids["<|endoftext|>"]
    # Generation assets, not the model config's EOS, define the stopping policy.
    expected_eos = {tokenizer.eos_token_id, token_ids["<|endofassistant|>"]}
    assert set(generation["eos_token_id"]) == expected_eos
    eos = _method(
        RUNTIME / "configs/model_config.py", "ModelConfig", "get_hf_eos_token_id", {}
    )
    assert (
        eos(
            SimpleNamespace(
                hf_config=config, hf_generation_config=SimpleNamespace(**generation)
            )
        )
        == expected_eos
    )
    template = (root / "chat_template.jinja").read_text().strip()
    assert tokenizer.chat_template.strip() == template
    if "chat_template" in tokenizer_config:
        assert tokenizer_config["chat_template"].strip() == template
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    rendered = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    assert rendered.endswith("<|endofassistant|>")
    assert (
        tokenizer.encode(rendered, add_special_tokens=False)[-1]
        == token_ids["<|endofassistant|>"]
    )
    assert config.quantization_config["quant_method"] == "fp8"
    assert config.quantization_config["fmt"] == "e4m3"
    assert config.quantization_config["weight_block_size"] == [128, 128]
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    headers = {}

    def tensor_header(key):
        shard = index[key]
        if shard not in headers:
            with (root / shard).open("rb") as handle:
                count = struct.unpack("<Q", handle.read(8))[0]
                headers[shard] = json.loads(handle.read(count))
        return headers[shard][key]

    for layer_type in ("full_attention", "sliding_attention"):
        layer_id = config.layer_types.index(layer_type)
        with torch.device("meta"):
            attn = adapter.Dots3NoteAttention(
                config,
                layer_id,
                SimpleNamespace(
                    attn=SimpleNamespace(tp_rank=0, tp_size=1, tp_group=None)
                ),
                is_nextn=False,
                quant_config=_Fp8Config(),
                prefix=f"model.layers.{layer_id}.self_attn",
            )
        for name, module in attn.named_modules():
            if not isinstance(module, _Linear):
                continue
            key = f"model.layers.{layer_id}.self_attn.{name}.weight"
            header = tensor_header(key)
            assert header["shape"] == list(module.weight.shape)
            assert header["dtype"] == "F8_E4M3"
            scale_shape = [(n + 127) // 128 for n in module.weight.shape]
            scale_header = tensor_header(f"{key}_scale_inv")
            assert scale_header["shape"] == scale_shape
            assert scale_header["dtype"] == "F32"


def test_native_import_opt_in():
    if os.environ.get("DOTS3_NOTE_NATIVE_TEST") != "1":
        pytest.skip(
            "Set DOTS3_NOTE_NATIVE_TEST=1 for the native dependency/import smoke test"
        )
    from tokenspeed.runtime.models.dots3_note import (
        Dot3NoteForCausalLM,
        Dots3NoteForCausalLM,
        EntryClass,
    )

    assert EntryClass == [Dot3NoteForCausalLM, Dots3NoteForCausalLM]


def test_hf_registration_and_backend_family(tmp_path, monkeypatch):
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config_cls = _config_class()
    exports = ast.parse((RUNTIME / "configs/__init__.py").read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tokenspeed.runtime.configs.dots3_note"
        and any(alias.name == "Dots3NoteConfig" for alias in node.names)
        for node in exports.body
    )
    hf_tree = ast.parse((RUNTIME / "utils/hf_transformers_utils.py").read_text())
    registry = next(
        node.value
        for node in hf_tree.body
        if isinstance(node, ast.AnnAssign) and node.target.id == "_CONFIG_REGISTRY"
    )
    entry = next(
        (key, value)
        for key, value in zip(registry.keys, registry.values)
        if isinstance(value, ast.Name) and value.id == "Dots3NoteConfig"
    )
    namespace = {"Dots3NoteConfig": config_cls}
    registration = eval(
        compile(
            ast.fix_missing_locations(
                ast.Expression(ast.Dict(keys=[entry[0]], values=[entry[1]]))
            ),
            "<HF registry>",
            "eval",
        ),
        namespace,
    )
    monkeypatch.setattr(
        CONFIG_MAPPING, "_extra_content", dict(CONFIG_MAPPING._extra_content)
    )
    for name, cls in registration.items():
        AutoConfig.register(name, cls)
    (tmp_path / "config.json").write_text(
        json.dumps(_config_values() | {"model_type": "dots3_note"})
    )
    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)
    assert isinstance(config, config_cls)
    assert config.architectures == ["Dot3NoteForCausalLM"]

    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "math": math,
        "IntEnum": IntEnum,
        "auto": auto,
        "resolve_architecture": lambda c: c.architectures[0],
    }
    _definitions(
        RUNTIME / "configs/model_config.py",
        [
            "AttentionArch",
            "_AttentionFamilySpec",
            "configure_dsa_attention",
            "_model_architectures",
            "_resolve_attention_family",
        ],
        namespace,
    )
    tree = ast.parse((RUNTIME / "configs/model_config.py").read_text())
    specs = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_ATTENTION_FAMILY_SPECS"
            for t in node.targets
        )
    )
    spec_node = next(
        node
        for node in specs.elts
        if any(
            kw.arg == "name"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "Dots3-note"
            for kw in node.keywords
        )
    )
    spec = eval(
        compile(ast.Expression(spec_node), "<attention family>", "eval"), namespace
    )
    namespace["_ATTENTION_FAMILY_SPECS"] = (spec,)
    assert namespace["_resolve_attention_family"](config, config) is spec
    assert spec.default_backend == "dots3_note"
    model = SimpleNamespace(hf_config=config, hf_text_config=config)
    spec.configure(model, SimpleNamespace())
    assert model.attention_arch == namespace["AttentionArch"].DSA
    assert (model.head_dim, model.kv_lora_rank, model.index_topk) == (192, 512, 2048)
    assert model.scaling == pytest.approx(192**-0.5)

    cli = ast.parse((RUNTIME / "utils/server_args.py").read_text())
    # Backend names are now validated by the registry, not argparse choices.
    option = next(
        node
        for node in ast.walk(cli)
        if isinstance(node, ast.Call)
        and any(
            isinstance(arg, ast.Constant) and arg.value == "--attention-backend"
            for arg in node.args
        )
    )
    assert not any(kw.arg == "choices" for kw in option.keywords)
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-backend")
    assert (
        parser.parse_args(["--attention-backend", "dots3_note"]).attention_backend
        == "dots3_note"
    )


@pytest.mark.parametrize("spelling", ["Dot3Note", "Dots3Note"])
@pytest.mark.parametrize("is_draft", [False, True])
def test_loader_uses_vlm_contract_only_for_target(monkeypatch, spelling, is_draft):
    from tokenspeed.runtime.configs.model_config import is_multimodal_model
    from tokenspeed.runtime.model_loader import loader
    from tokenspeed.runtime.models import dots3_note, dots3_note_nextn
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3ForCausalLM

    architecture = f"{spelling}ForCausalLM" + ("NextN" if is_draft else "")
    module = dots3_note_nextn if is_draft else dots3_note
    model_cls = getattr(module, architecture)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="dots3_note"),
        mapping=object(),
        is_multimodal=is_multimodal_model([architecture]),
        is_multimodal_active=False,
        mm_attention_backend=None,
    )
    assert config.is_multimodal is (not is_draft)
    init = create_autospec(DeepseekV3ForCausalLM.__init__, return_value=None)
    monkeypatch.setattr(DeepseekV3ForCausalLM, "__init__", init)
    monkeypatch.setattr(
        loader, "get_model_architecture", lambda c: (model_cls, architecture)
    )
    quant_config = object()
    monkeypatch.setattr(loader, "_get_quantization_config", lambda *a: quant_config)
    loader._initialize_model(config, None)
    assert init.call_args.kwargs["config"] is config.hf_config
    assert init.call_args.kwargs["mapping"] is config.mapping
    assert init.call_args.kwargs["quant_config"] is quant_config
    if is_draft:
        with pytest.raises(TypeError, match="is_multimodal_active"):
            model_cls(config.hf_config, config.mapping, is_multimodal_active=False)
    else:
        with pytest.raises(TypeError, match="is_multimodal_active"):
            model_cls(config.hf_config, config.mapping)
        config.is_multimodal_active = True
        with pytest.raises(ValueError, match="--language-model-only"):
            loader._initialize_model(config, None)
        config.is_multimodal_active = False
        config.mm_attention_backend = "triton_attn"
        with pytest.raises(ValueError, match="multimodal attention backend"):
            loader._initialize_model(config, None)


@pytest.mark.parametrize(
    "architecture", ["Dot3NoteForCausalLM", "Dots3NoteForCausalLM"]
)
@pytest.mark.parametrize("is_draft", [False, True])
@pytest.mark.parametrize("language_model_only", [False, True])
@pytest.mark.parametrize("encoders", [{}, {"vision_config": {}, "audio_config": {}}])
def test_native_model_config_preserves_source_geometry(
    tmp_path, architecture, is_draft, language_model_only, encoders
):
    from tokenspeed.runtime.configs.model_config import AttentionArch, ModelConfig
    from tokenspeed.runtime.utils.server_args import ServerArgs

    values = (
        _config_values()
        | encoders
        | {
            "model_type": "dots3_note",
            "architectures": [architecture],
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(values))
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 151668})
    )
    args = ServerArgs(
        model=str(tmp_path),
        prefix_granularity=64,
        speculative_algorithm="MTP" if is_draft else None,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        language_model_only=language_model_only,
    )
    kwargs = dict(
        trust_remote_code=False,
        model_override_args="{}",
        is_draft_worker=is_draft,
        server_args=args,
    )
    if not is_draft and not language_model_only:
        with pytest.raises(ValueError, match="use --language-model-only"):
            ModelConfig(str(tmp_path), **kwargs)
        return
    config = ModelConfig(str(tmp_path), **kwargs)
    assert config.is_multimodal is (not is_draft)
    assert not config.is_multimodal_active
    assert config.hf_config.architectures == [
        architecture + ("NextN" if is_draft else "")
    ]
    assert config.hf_text_config.num_hidden_layers == 46
    assert len(config.hf_text_config.layer_types) == 46
    assert config.num_attention_layers == (1 if is_draft else 46)
    assert config.attention_arch == (
        AttentionArch.MLA if is_draft else AttentionArch.DSA
    )
    assert (
        config.num_attention_heads
        == config.num_key_value_heads
        == (64 if is_draft else 128)
    )
    assert config.kv_lora_rank == (1024 if is_draft else 512)
    assert config.head_dim == (256 if is_draft else 192)
    assert (
        args.drafter_attention_backend if is_draft else args.attention_backend
    ) == "dots3_note"


@pytest.mark.parametrize("generation_ids", [[151643, 151668], 151668, [7, 8]])
def test_dots3_eos_generation_assets_are_authoritative(generation_ids):
    method = _method(
        RUNTIME / "configs/model_config.py", "ModelConfig", "get_hf_eos_token_id", {}
    )
    model = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="dots3_note", eos_token_id=151645),
        hf_generation_config=SimpleNamespace(eos_token_id=generation_ids),
    )
    expected = (
        {generation_ids} if isinstance(generation_ids, int) else set(generation_ids)
    )
    assert method(model) == expected
    model.hf_config.model_type = "deepseek_v3"
    assert method(model) == expected | {151645}


@pytest.mark.parametrize(
    "generation",
    [None, SimpleNamespace(eos_token_id=None), SimpleNamespace(eos_token_id=[])],
)
def test_dots3_eos_missing_generation_assets_fail(generation):
    method = _method(
        RUNTIME / "configs/model_config.py", "ModelConfig", "get_hf_eos_token_id", {}
    )
    model = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="dots3_note", eos_token_id=151645),
        hf_generation_config=generation,
    )
    with pytest.raises(ValueError, match="generation_config.json"):
        method(model)
