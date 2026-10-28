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

"""HY V4 checkpoint layouts and model-specific numerical boundaries."""

from __future__ import annotations

import itertools
import json
import os
import sys

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.hy_v4_config import HYV4Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.rotary_embedding import RotaryEmbedding, get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.models.hy_v4 import (
    HYV4Attention,
    HYV4HCBoundary,
    HYV4RotaryEmbedding,
    check_hyv4_quant_config,
    load_hyv4_weights,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")


def _config(**overrides) -> HYV4Config:
    values = dict(
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
        mlp_layer_types=["dense"],
        indexer_types=["full"],
        gating_type="elementwise",
    )
    values.update(overrides)
    return HYV4Config(**values)


@pytest.fixture
def runtime_settings():
    previous = global_server_args_dict.copy()
    previous_dtype = torch.get_default_dtype()
    global_server_args_dict.update(
        tp_batch_invariant="none",
        numerics="auto",
        dense_gemm_backend="auto",
        attention_backend="dsa",
    )
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(previous)
        torch.set_default_dtype(previous_dtype)


def test_config_json_roundtrip_preserves_attention_and_layer_schedule():
    config = _config(
        num_hidden_layers=3,
        mlp_layer_types=["dense", "sparse", "sparse"],
        indexer_types=["full", "shared", "full"],
        layer_types=["sparse", "sparse_attention", "deepseek_sparse_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 5e6},
    )
    restored = HYV4Config.from_dict(json.loads(config.to_json_string()))
    assert restored.mlp_layer_types == ["dense", "sparse", "sparse"]
    assert restored.indexer_types == ["full", "shared", "full"]
    assert restored.layer_types == ["deepseek_sparse_attention"] * 3
    assert restored.first_k_dense_replace == 1
    assert restored.qk_head_dim == 8
    assert restored.head_dim == restored.qk_rope_head_dim == 4
    assert restored.rope_theta == 5e6
    assert restored.rope_scaling is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_hidden_layers": 2, "indexer_types": ["full"]},
        {"num_hidden_layers": 2, "mlp_layer_types": ["dense"]},
        {"layer_types": ["full_attention"]},
        {"head_dim": 8},
        {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e7}},
        {"rope_scaling": {"type": "linear", "factor": 2.0}},
        {
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e7},
            "rope_scaling": {"type": "default", "rope_theta": 5e6},
        },
        {
            "num_hidden_layers": 3,
            "mlp_layer_types": ["dense", "sparse", "dense"],
            "indexer_types": ["full", "shared", "full"],
        },
    ],
)
def test_config_rejects_layouts_that_cannot_run_as_hyv4(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_legacy_plain_rope_survives_config_roundtrip():
    config = _config(rope_scaling={"type": "default", "rope_theta": 5e6})
    restored = HYV4Config.from_dict(config.to_dict())
    assert restored.rope_theta == 5e6
    assert restored.rope_parameters["rope_type"] == "default"
    assert restored.rope_scaling is None


class _CheckpointOwner(nn.Module):
    """Small module tree with production linear and iHC parameter loaders."""

    def __init__(self, *, shared_head: bool, draft_prefix: bool):
        super().__init__()
        self.config = _config()
        self.mapping = Mapping(rank=0, world_size=1)
        self.quant_config = None
        qkv = ReplicatedLinear(16, 16, bias=False, params_dtype=torch.bfloat16)
        gate_up = MergedColumnParallelLinear(
            16,
            [8, 8],
            bias=False,
            params_dtype=torch.bfloat16,
            tp_rank=0,
            tp_size=1,
            tp_group=(0,),
        )
        layer = nn.ModuleDict(
            {
                "self_attn": nn.ModuleDict({"fused_qkv_a_proj_with_mqa": qkv}),
                "mlp": nn.ModuleDict({"gate_up_proj": gate_up}),
                "hc_attn_layer": HYV4HCBoundary(self.config, is_head=False),
            }
        )
        self.model = nn.ModuleDict(
            {"decoder": layer} if draft_prefix else {"layers": nn.ModuleList([layer])}
        )
        self.lm_head = (
            nn.Linear(16, 32, bias=False, dtype=torch.bfloat16) if shared_head else None
        )


def _checkpoint(owner, *, prefix):
    expected = {}
    weights = []
    for i, (name, parameter) in enumerate(owner.named_parameters()):
        if name.startswith("lm_head."):
            continue
        value = (
            torch.arange(parameter.numel(), dtype=torch.float32)
            .reshape(parameter.shape)
            .div(17)
            .add(i / 31)
            .to(parameter.dtype)
        )
        expected[name] = value
        suffix = name.split(".self_attn.")[-1]
        if suffix == "fused_qkv_a_proj_with_mqa.weight":
            weights.extend(
                [
                    (f"{prefix}.self_attn.q_a_proj.weight", value[:8]),
                    (f"{prefix}.self_attn.kv_a_proj_with_mqa.weight", value[8:]),
                ]
            )
        elif name.endswith(".mlp.gate_up_proj.weight"):
            weights.extend(
                [
                    (f"{prefix}.mlp.gate_proj.weight", value[:8]),
                    (f"{prefix}.mlp.up_proj.weight", value[8:]),
                ]
            )
        else:
            leaf = name.rsplit(".", 1)[-1]
            weights.append((f"{prefix}.hc_attn_layer.hc_pre.{leaf}", value))
    return weights, expected


@pytest.mark.parametrize("shard_order", list(itertools.permutations(range(4))))
def test_checkpoint_fused_shards_load_exact_values_in_any_order(
    shard_order, runtime_settings
):
    owner = _CheckpointOwner(shared_head=False, draft_prefix=False)
    weights, expected = _checkpoint(owner, prefix="model.layers.0")
    ordered = [weights[i] for i in shard_order] + weights[4:]
    load_hyv4_weights(owner, ordered, module_prefix="model", checkpoint_prefix="model")
    for name, value in expected.items():
        torch.testing.assert_close(owner.get_parameter(name), value, rtol=0, atol=0)
    # Values not representable in BF16 must survive in the iHC master parameters.
    fn = owner.get_parameter("model.layers.0.hc_attn_layer.hc_fn")
    assert fn.dtype == torch.float32
    assert not torch.equal(fn, fn.bfloat16().float())


@pytest.mark.parametrize("missing", range(4))
def test_checkpoint_rejects_incomplete_fused_projection(missing, runtime_settings):
    owner = _CheckpointOwner(shared_head=False, draft_prefix=False)
    weights, _ = _checkpoint(owner, prefix="model.layers.0")
    del weights[missing]
    with pytest.raises(RuntimeError, match="unloaded"):
        load_hyv4_weights(
            owner, weights, module_prefix="model", checkpoint_prefix="model"
        )


@pytest.mark.parametrize("duplicate", range(4))
def test_checkpoint_rejects_duplicate_fused_shards(duplicate, runtime_settings):
    owner = _CheckpointOwner(shared_head=False, draft_prefix=False)
    weights, _ = _checkpoint(owner, prefix="model.layers.0")
    weights.insert(duplicate, weights[duplicate])
    with pytest.raises(RuntimeError, match="Duplicate fused shard"):
        load_hyv4_weights(
            owner, weights, module_prefix="model", checkpoint_prefix="model"
        )


def test_checkpoint_prefix_remapping_and_shared_head(runtime_settings):
    owner = _CheckpointOwner(shared_head=True, draft_prefix=True)
    weights, expected = _checkpoint(owner, prefix="model.mtp_layers.0")
    owner.lm_head.weight.data.fill_(0.375)
    shared_before = owner.lm_head.weight.clone()
    load_hyv4_weights(
        owner,
        reversed(weights),
        module_prefix="model.decoder",
        checkpoint_prefix="model.mtp_layers.0",
        shared_params=("lm_head.",),
    )
    for name, value in expected.items():
        torch.testing.assert_close(owner.get_parameter(name), value, rtol=0, atol=0)
    torch.testing.assert_close(owner.lm_head.weight, shared_before, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        load_hyv4_weights(
            owner,
            weights,
            module_prefix="model.decoder",
            checkpoint_prefix="model.mtp_layers.0",
        )


class _MtpProjectionOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _config()
        self.mapping = Mapping(rank=0, world_size=1)
        self.quant_config = None
        self.model = nn.ModuleDict(
            {"eh_proj": nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)}
        )


def test_loader_rejects_fp8_for_unquantized_mtp_projection(runtime_settings):
    owner = _MtpProjectionOwner()
    weight = torch.ones((16, 32), dtype=torch.float32).to(torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="Unexpected FP8"):
        load_hyv4_weights(
            owner,
            [("model.eh_proj.weight", weight)],
            module_prefix="model",
            checkpoint_prefix="model",
        )


@pytest.mark.parametrize("suffix", ["weight_scale", "weight_scale_inv"])
def test_loader_rejects_unconsumed_quantization_scales(suffix, runtime_settings):
    owner = _MtpProjectionOwner()
    with pytest.raises(RuntimeError, match="Unmatched HY V4 quantization scales"):
        load_hyv4_weights(
            owner,
            [
                ("model.eh_proj.weight", torch.ones(16, 32, dtype=torch.bfloat16)),
                (
                    f"model.eh_proj.{suffix}",
                    torch.full((16, 1), 127, dtype=torch.uint8),
                ),
            ],
            module_prefix="model",
            checkpoint_prefix="model",
        )


def test_full_bf16_checkpoint_is_refused_before_model_allocation():
    with pytest.raises(ValueError, match="ModelOpt MXFP8 checkpoints only"):
        check_hyv4_quant_config(None)


def test_rope_coefficients_keep_hyv4_rounding_without_mutating_generic_cache():
    kwargs = dict(
        head_size=4,
        rotary_dim=4,
        max_position=32,
        base=1e7,
        is_neox_style=False,
        dtype=torch.bfloat16,
    )
    with torch.device("cpu"):
        generic = get_rope(**kwargs)
        original = generic.cos_sin_cache.clone()
        rotary = HYV4RotaryEmbedding(4, 4, 32, 1e7, False, torch.bfloat16)
        fresh_reference = RotaryEmbedding(4, 4, 32, 1e7, False, torch.bfloat16)
    expected = fresh_reference.cos_sin_cache.bfloat16().float()
    assert rotary.cos_sin_cache.dtype == torch.float32
    torch.testing.assert_close(rotary.cos_sin_cache, expected, rtol=0, atol=0)
    assert not torch.equal(expected, fresh_reference.cos_sin_cache)
    torch.testing.assert_close(generic.cos_sin_cache, original, rtol=0, atol=0)


def test_sink_loads_local_heads_and_binding_sees_loaded_values(runtime_settings):
    with torch.device("cpu"):
        attention = HYV4Attention(
            _config(), Mapping(rank=1, world_size=2), layer_id=0, is_nextn=False
        )
    checkpoint = torch.tensor([0.1234567, 0.7654321], dtype=torch.float32)
    parameter = attention.learnable_sink_param
    parameter.weight_loader(parameter, checkpoint)
    attention.bind_learnable_sink()
    torch.testing.assert_close(
        attention.attn_mqa.attn_sink, checkpoint[1:], rtol=0, atol=0
    )
    with torch.no_grad():
        parameter.add_(0.25)
    torch.testing.assert_close(
        attention.attn_mqa.attn_sink, checkpoint[1:] + 0.25, rtol=0, atol=0
    )


def test_lm_head_keeps_fp32_logits_with_bf16_weight_storage(runtime_settings):
    config = _config(head_dtype="float32")
    processor = LogitsProcessor(config, dp_lm_head_tp=False)
    head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        params_dtype=torch.bfloat16,
        padding_size=1,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    )
    hidden = torch.arange(48, dtype=torch.float32).view(3, 16).div(17).bfloat16()
    with torch.no_grad():
        values = torch.arange(head.weight.numel(), dtype=torch.float32)
        head.weight.copy_((values.remainder(17) / 37).view_as(head.weight))
    expected = hidden.float() @ head.weight.float().T
    output = processor._get_logits(hidden, head, None, require_full_vocab=True)
    assert output.dtype == torch.float32
    assert head.weight.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert not torch.equal(output, expected.bfloat16().float())


@pytest.fixture(params=[True, False], ids=["reduced-allowed", "reduced-disabled"])
def lm_head_precision(request):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_reduction = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = request.param
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            previous_reduction
        )


def _cuda_lm_head(hidden, vocab):
    config = _config(hidden_size=hidden, vocab_size=vocab)
    processor = LogitsProcessor(config, dp_lm_head_tp=False)
    with torch.device("cuda"):
        head = ParallelLMHead(
            vocab,
            hidden,
            params_dtype=torch.bfloat16,
            padding_size=1,
            tp_rank=0,
            tp_size=1,
            tp_group=(0,),
        )
    return processor, head


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA LM head")
@pytest.mark.parametrize("rows,vocab", [(1, 3776), (4, 3776), (32, 3776), (1, 120832)])
def test_lm_head_gpu_fp32_output_matches_fp32_operand_reference(
    rows, vocab, runtime_settings, lm_head_precision
):
    # 3776 is HY4's TP32 vocabulary shard. Only M=1 uses the full vocabulary.
    processor, head = _cuda_lm_head(6144, vocab)
    generator = torch.Generator(device="cuda").manual_seed(20261028)
    with torch.no_grad():
        head.weight.normal_(0, 0.03125, generator=generator)
    hidden = torch.randn(
        rows, 6144, dtype=torch.bfloat16, device="cuda", generator=generator
    ).mul_(0.125)
    reference = hidden.float() @ head.weight.float().T
    actual = processor._get_logits(hidden, head, None, require_full_vocab=True)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    # Both paths accumulate in FP32, but their reduction orders may differ.
    delta = (actual - reference).double()
    relative_error = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
        reference.double()
    )
    assert relative_error.item() < 1e-5
    assert delta.abs().max().item() < 5e-5
    assert not torch.equal(actual, actual.bfloat16().float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA LM head")
@pytest.mark.parametrize("hidden_size", [6144, 32768])
def test_lm_head_gpu_preserves_small_terms_across_large_cancellation(
    hidden_size, runtime_settings, lm_head_precision
):
    processor, head = _cuda_lm_head(hidden_size, 17)
    hidden = torch.ones(4, hidden_size, dtype=torch.bfloat16, device="cuda")
    hidden[:, 0] = 65536
    hidden[:, -1] = -65536
    hidden.mul_(torch.tensor([1, 2, 0.5, -1], device="cuda")[:, None])
    powers = torch.arange(17, device="cuda").remainder(7) - 3
    with torch.no_grad():
        head.weight.copy_(torch.exp2(powers.float())[:, None].expand_as(head.weight))
    # All operands and all possible FP32 partial sums are exactly representable.
    # A BF16 partial reduction can lose the small terms beside +/-65536.
    reference = hidden.cpu().double() @ head.weight.cpu().double().T
    actual = processor._get_logits(hidden, head, None, require_full_vocab=True)
    torch.testing.assert_close(actual.cpu().double(), reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA LM head")
def test_lm_head_fp32_cuda_graph_reads_changed_bf16_inputs(
    runtime_settings, lm_head_precision
):
    processor, head = _cuda_lm_head(6144, 3776)
    generator = torch.Generator(device="cuda").manual_seed(20261028)
    with torch.no_grad():
        head.weight.normal_(0, 0.03125, generator=generator)
    hidden = torch.randn(
        4, 6144, dtype=torch.bfloat16, device="cuda", generator=generator
    ).mul_(0.125)
    for _ in range(3):
        processor._get_logits(hidden, head, None, require_full_vocab=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = processor._get_logits(hidden, head, None, require_full_vocab=True)
    graph.replay()
    first = output.clone()
    hidden.mul_(0.5)
    graph.replay()
    eager = processor._get_logits(hidden, head, None, require_full_vocab=True)
    torch.testing.assert_close(output, eager, rtol=0, atol=0)
    assert not torch.equal(first, output)
    assert output.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real linear gate requires CUDA"
)
@pytest.mark.parametrize("gating_type", ["elementwise", "headwise"])
def test_attention_gate_rounds_sigmoid_before_bf16_product(
    gating_type, runtime_settings
):
    with torch.device("cuda"):
        attention = HYV4Attention(
            _config(gating_type=gating_type),
            Mapping(rank=0, world_size=1),
            layer_id=0,
            is_nextn=False,
        )
    weight = attention.linear_gate.weight
    with torch.no_grad():
        weight.zero_()
        indices = torch.arange(weight.shape[0], device="cuda")
        weight[indices, indices] = 1
    hidden = torch.linspace(-2, 2, 48, device="cuda").reshape(3, 16).bfloat16()
    output = torch.linspace(-2, 2, 24, device="cuda").reshape(3, 8).bfloat16()
    hidden[0, 0], output[0, 0] = 0.5, 1.5
    gate = hidden[:, : weight.shape[0]]
    if gating_type == "headwise":
        expected = (output.view(3, 2, 4) * gate.sigmoid().unsqueeze(-1)).view(3, 8)
        fused_fp32 = (
            (output.float().view(3, 2, 4) * gate.float().sigmoid().unsqueeze(-1))
            .view(3, 8)
            .bfloat16()
        )
    else:
        expected = output * gate.sigmoid()
        fused_fp32 = (output.float() * gate.float().sigmoid()).bfloat16()
    assert expected[0, 0] != fused_fp32[0, 0]
    actual = attention._apply_output_gate(output.clone(), hidden)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
