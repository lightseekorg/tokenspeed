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

from __future__ import annotations

import math
import os
import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.linear import ColumnParallelLinear, RowParallelLinear

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed_kernel.ops.residual import residual_gate_coefficients

from tokenspeed.runtime.configs.model_config import is_audio_model, is_multimodal_model
from tokenspeed.runtime.configs.muse_spark_config import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    MuseSparkConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    hybrid_slab_group_size,
    layer_group_ids,
)
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.layers.quantization.utils import should_exclude_quant_module
from tokenspeed.runtime.models import muse_spark as module
from tokenspeed.runtime.models.muse_spark import (
    RESIDUAL_GATE_EPS,
    MuseSparkAttention,
    MuseSparkDecoderLayer,
    MuseSparkForConditionalGeneration,
    MuseSparkMoE,
    QKVOGateParallelLinear,
    _widen_modelopt_exclusions,
    exclude_fully_excluded_fused_modules,
    resolve_parameter_name,
    resolve_per_expert_tensor,
    restore_rotary_pairs,
    shard_expert_projection,
)
from tokenspeed.runtime.models.muse_spark_vision import (
    FULL_ATTENTION as VISION_FULL_ATTENTION,
)
from tokenspeed.runtime.models.muse_spark_vision import (
    WINDOW_ATTENTION,
    apply_complex_rope,
    downsample_pixel_shuffle,
    pixel_shuffle_permutation,
    sparse_permutation,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)


@pytest.fixture
def bf16_default_dtype():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(previous_dtype)


def _layer_types(num_hidden_layers: int) -> list[str]:
    """Three sliding layers per full one, counted back from the last layer."""
    return [
        (
            FULL_ATTENTION
            if (num_hidden_layers - layer_id - 1) % 4 == 0
            else SLIDING_ATTENTION
        )
        for layer_id in range(num_hidden_layers)
    ]


def _text_config(num_hidden_layers: int) -> dict:
    layer_types = _layer_types(num_hidden_layers)
    return {
        "vocab_size": 256,
        "hidden_size": 256,
        "head_dim": 32,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": num_hidden_layers,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "final_logit_softcapping": 20.0,
        "qk_scale_factor": 5.4730064863875,
        "output_multiplier": 0.125,
        "residual_gate_temperature": 0.3,
        "sliding_window": 128,
        "layer_types": layer_types,
        "layer_rope_theta": [
            0.0 if layer_type == FULL_ATTENTION else 500000.0
            for layer_type in layer_types
        ],
        "mlp_layer_types": ["sparse"] * num_hidden_layers,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "moe_hidden_size": 128,
        "moe_intermediate_size": 64,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 0,
        "routed_scaling_factor": 1.0,
    }


def _vision_config(num_hidden_layers: int, hidden_size: int) -> dict:
    layer_types = [WINDOW_ATTENTION] * num_hidden_layers
    for layer_id in range(3, num_hidden_layers, 4):
        layer_types[layer_id] = VISION_FULL_ATTENTION
    layer_types[-1] = VISION_FULL_ATTENTION
    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "layer_types": layer_types,
        "layer_norm_eps": 1e-5,
        "merge_size": 2,
        "patch_size": 2,
        "patch_temporal": 2,
        "pos_emb_height": 2,
        "pos_emb_width": 2,
        "max_position_embeddings": 4,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
        "hidden_act": "gelu",
    }


def _audio_config(hidden_size: int) -> dict:
    return {
        "conv_kernel_size": 7,
        "depths": [1, 1, 1, 1],
        "hidden_act": "gelu",
        "hidden_sizes": [8, 16, 32, 64],
        "mlp_ratio": 4.0,
        "num_hidden_layers": 4,
        "out_hidden_size": hidden_size,
        "patch_sizes": [16, 6, 5, 4],
        "rms_norm_eps": 1e-6,
    }


def _config(num_hidden_layers: int) -> MuseSparkConfig:
    vision_hidden_size = 32
    return MuseSparkConfig(
        text_config=_text_config(num_hidden_layers),
        vision_config=_vision_config(1, vision_hidden_size),
        audio_config=_audio_config(256),
        image_token_id=200,
        video_token_id=201,
        audio_token_id=202,
        out_hidden_size=vision_hidden_size * 4,
        projector_hidden_size=64,
        projector_hidden_act="gelu",
    )


def _tiny_config() -> MuseSparkConfig:
    return _config(1)


def _single_rank_mapping() -> Mapping:
    return Mapping(
        rank=0,
        world_size=1,
        attn_tp_size=1,
        attn_kvp_size=1,
        attn_qcp_size=1,
        attn_dp_size=1,
        dense_tp_size=1,
        dense_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=1,
        linear_attn_tp_size=1,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=1,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def _tiny_model() -> MuseSparkForConditionalGeneration:
    with torch.device("cuda"):
        return MuseSparkForConditionalGeneration(
            config=_tiny_config(),
            mapping=_single_rank_mapping(),
            quant_config=None,
            is_multimodal_active=True,
            mm_attention_backend="triton_attn",
        )


def _stub(cls: type) -> torch.nn.Module:
    instance = cls.__new__(cls)
    torch.nn.Module.__init__(instance)
    return instance


def _rmsnorm(values: torch.Tensor, eps: float) -> torch.Tensor:
    return values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)


def test_muse_spark_architecture_is_multimodal() -> None:
    assert is_multimodal_model(["MuseSparkForConditionalGeneration"])


def test_muse_spark_architecture_is_an_audio_model() -> None:
    assert is_audio_model(["MuseSparkForConditionalGeneration"])


def test_muse_spark_config_groups_the_attention_pattern_into_caches() -> None:
    config = _config(62)
    s, f = SLIDING_ATTENTION, FULL_ATTENTION
    assert config.layer_types[:6] == [s, f, s, s, s, f]
    assert config.sliding_window[:6] == (128, None, 128, 128, 128, None)
    uses_rope = [config.uses_rope(i) for i in range(6)]
    assert uses_rope == [layer_type == s for layer_type in config.layer_types[:6]]
    windows = config.sliding_window
    group_ids = layer_group_ids(
        layer_types=config.cache_layer_types, sliding_window_tokens=windows
    )
    assert group_ids[:6] == [f"{s}_0", f, f"{s}_1", f"{s}_2", f"{s}_0", f]
    counts = {g: group_ids.count(g) for g in set(group_ids)}
    assert counts == {f: 16, f"{s}_0": 16, f"{s}_1": 15, f"{s}_2": 15}
    slab = hybrid_slab_group_size(
        config.cache_layer_types, sliding_window_tokens=windows
    )
    assert slab == 16
    assert config.architectures == ["MuseSparkForConditionalGeneration"]
    assert config.model_type == "muse_spark"
    assert (config.num_local_experts, config.num_experts_per_tok) == (4, 2)
    assert config.moe_intermediate_size == 64
    assert config.final_logit_softcapping == 20.0
    assert config.output_multiplier == 0.125


def test_config_rejects_a_rope_schedule_the_windows_contradict() -> None:
    text = _text_config(8)
    text["layer_rope_theta"][3] = 500000.0
    with pytest.raises(ValueError, match="RoPE base"):
        MuseSparkConfig(
            text_config=text,
            vision_config=_vision_config(1, 32),
            audio_config=_audio_config(256),
            image_token_id=200,
            video_token_id=201,
            audio_token_id=202,
            out_hidden_size=128,
            projector_hidden_size=64,
            projector_hidden_act="gelu",
        )


class _AddLayer(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.input_layernorm = f"norm{value}"

    def forward(
        self,
        positions,
        hidden_states,
        ctx,
        attention_input,
        next_input_layernorm,
        owned_global_rows,
    ):
        self.received_attention_input = attention_input
        self.received_next_norm = next_input_layernorm
        # Emit a marker the model must hand to the next layer untouched.
        return hidden_states + self.value, hidden_states * self.value


def test_forward_threads_attention_input_rows_between_layers() -> None:
    model = _stub(MuseSparkForConditionalGeneration.model_cls)
    model.layers = torch.nn.ModuleList(_AddLayer(float(i + 1)) for i in range(3))
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    model._embed_hidden = lambda input_ids, input_embeds: input_embeds
    model.embed_tokens = SimpleNamespace(weight=torch.empty(0, dtype=torch.bfloat16))
    model.norm = lambda value, out_dtype: -value.to(out_dtype)
    ctx = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    out, aux = model.forward(torch.tensor([0, 1]), torch.tensor([0, 1]), ctx, hidden)
    assert aux is None
    torch.testing.assert_close(out, -(hidden + 1.0 + 2.0 + 3.0).to(torch.bfloat16))
    # The first layer normalizes for itself; later ones reuse the previous boundary's rows.
    assert model.layers[0].received_attention_input is None
    torch.testing.assert_close(model.layers[1].received_attention_input, hidden * 1.0)
    torch.testing.assert_close(
        model.layers[2].received_attention_input, (hidden + 1.0) * 2.0
    )
    # Each boundary gets the next layer's input norm; the last layer gets none.
    assert [layer.received_next_norm for layer in model.layers] == [
        "norm2.0",
        "norm3.0",
        None,
    ]


@requires_gpu
def test_muse_mxfp4_honors_quark_expert_exclusions(
    bf16_default_dtype,
) -> None:
    quant_config = Mxfp4Config(
        ignored_layers=["model.layers.0.mlp.experts"],
        is_checkpoint_mxfp4_serialized=True,
        is_w4a8_fp8=False,
        use_dynamic_mxfp4_activations=True,
        quant_method="quark",
    )
    with torch.device("cuda"):
        kept = MuseSparkMoE(
            config=_tiny_config(),
            mapping=_single_rank_mapping(),
            quant_config=quant_config,
            layer_id=0,
            prefix="model.layers.0.mlp",
            alt_stream=None,
        )
        quantized = MuseSparkMoE(
            config=_tiny_config(),
            mapping=_single_rank_mapping(),
            quant_config=quant_config,
            layer_id=1,
            prefix="model.layers.1.mlp",
            alt_stream=None,
        )

    assert kept.quantized_experts is False
    assert kept.expert_quant_kind == "unquant"
    assert kept.experts.w13_weight.dtype == torch.bfloat16
    assert quantized.quantized_experts is True
    assert quantized.expert_quant_kind == "mxfp4"
    assert quantized.experts.w13_weight.dtype == torch.uint8
    assert quantized.experts.plan["solution"] == "triton"
    assert quantized.experts.supports_deferred_finalize is True
    assert quantized.experts.deferred_output_is_token_order is False


@requires_gpu
def test_muse_spark_moe_matches_torch_reference(bf16_default_dtype) -> None:
    with torch.device("cuda"):
        module_ = MuseSparkMoE(
            config=_tiny_config(),
            mapping=_single_rank_mapping(),
            quant_config=None,
            layer_id=0,
            prefix="mlp",
            alt_stream=None,
        )
    torch.manual_seed(21)
    for parameter in module_.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)
    module_.gate.process_weights_after_loading(module_.gate)
    hidden = torch.randn(5, 256, dtype=torch.bfloat16, device="cuda")
    router_input = hidden.float()
    output, sentinel = module_(
        hidden, router_input, hidden.shape[0], hidden.shape[0], None, project=True
    )
    assert sentinel is None

    logits = router_input @ module_.gate.weight
    logits *= module_.gate.multiplier
    scores = logits.sigmoid()
    ids = torch.topk(
        scores + module_.gate.e_score_correction_bias, 2, dim=-1, sorted=False
    ).indices
    route_weights = scores.gather(1, ids)
    route_weights /= route_weights.sum(dim=-1, keepdim=True)
    compressed = (hidden.float() @ module_.pre_expert_proj.weight.float().T).to(
        torch.bfloat16
    )
    # The Triton suite pins the norm; a one-ulp difference would dominate here.
    compressed = module_.pre_expert_norm(compressed, compressed.dtype)
    routes = torch.zeros(5, 2, 128, dtype=torch.bfloat16, device="cuda")
    for token in range(5):
        for slot in range(2):
            expert = int(ids[token, slot].item())
            gate_up = (
                compressed[token].float() @ module_.experts.w13_weight[expert].float().T
            )
            gate, up = gate_up.chunk(2)
            activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            routes[token, slot] = (
                activated.float() @ module_.experts.w2_weight[expert].float().T
            ).to(torch.bfloat16)
    scaled_routes = routes.float() * (module_.post_expert_norm.weight + 1.0).float()
    normalized_routes = _rmsnorm(scaled_routes, module_.post_expert_norm.eps)
    combined = (normalized_routes * route_weights[..., None]).sum(dim=1)
    reference = (
        combined.to(torch.bfloat16).float() @ module_.post_expert_proj.weight.float().T
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)


def test_tensor_parallel_expert_shards_reconstruct_full_routes() -> None:
    generator = torch.Generator().manual_seed(173)
    hidden, intermediate, experts, ranks = 32, 64, 3, 8
    x = torch.randn(5, hidden, dtype=torch.float64, generator=generator)
    w13 = torch.randn(
        experts, 2 * intermediate, hidden, dtype=torch.float64, generator=generator
    )
    w2 = torch.randn(
        experts, hidden, intermediate, dtype=torch.float64, generator=generator
    )
    gate, up = torch.einsum("mh,eih->mei", x, w13).chunk(2, dim=-1)
    expected = torch.einsum("mei,ehi->meh", torch.nn.functional.silu(gate) * up, w2)
    actual = torch.zeros_like(expected)
    for rank in range(ranks):
        local_w13 = torch.empty(experts, 2 * intermediate // ranks, hidden)
        local_w2 = torch.empty(experts, hidden, intermediate // ranks)
        gate_up = shard_expert_projection(
            w13, local_w13, rank, ranks, gate_up=True, fused_gate_up=True
        )
        down = shard_expert_projection(
            w2, local_w2, rank, ranks, gate_up=False, fused_gate_up=False
        )
        gate, up = torch.einsum("mh,eih->mei", x, gate_up).chunk(2, dim=-1)
        actual += torch.einsum(
            "mei,ehi->meh", torch.nn.functional.silu(gate) * up, down
        )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@requires_gpu
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "mxfp4"])
def test_tensor_parallel_checkpoint_loading_preserves_gate_up_pairing(
    bf16_default_dtype, monkeypatch: pytest.MonkeyPatch, quantized: bool
) -> None:
    # Weight loading needs no active collective; only its absent process group
    # is substituted. The real model and its parameter buffers are constructed.
    monkeypatch.setattr(
        module.process_group_manager, "get_device_process_group", lambda group: None
    )
    rank, ranks, intermediate = 7, 8, 4096
    config = _tiny_config()
    config.moe_intermediate_size = intermediate
    mapping = Mapping(
        rank=rank,
        world_size=ranks,
        attn_tp_size=ranks,
        attn_dp_size=1,
        dense_tp_size=ranks,
        dense_dp_size=1,
        moe_tp_size=ranks,
        moe_ep_size=1,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=ranks,
        linear_attn_tp_size=ranks,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=ranks,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )
    quant_config = (
        Mxfp4Config(
            is_checkpoint_mxfp4_serialized=True,
            use_dynamic_mxfp4_activations=True,
        )
        if quantized
        else None
    )
    with torch.device("cuda"):
        model = MuseSparkForConditionalGeneration(
            config,
            mapping,
            quant_config,
            is_multimodal_active=False,
            mm_attention_backend=None,
        )
    moe = model.model.layers[0].mlp
    if quantized and current_platform().is_cdna4:
        assert (
            moe.experts.plan["apply_kernel_name"]
            == "gluon_mxfp4_a4w4_ep_deferred_precomputed_moe_apply"
        )
        assert moe.experts.deferred_output_is_token_order is True
    local_intermediate = intermediate // ranks
    weight_divisor = 2 if quantized else 1
    specifications = [
        ("gate_up_proj", "w13_weight", (4, 2 * intermediate, 128 // weight_divisor)),
        ("down_proj", "w2_weight", (4, 128, intermediate // weight_divisor)),
    ]
    if quantized:
        specifications += [
            ("gate_up_proj_scale", "w13_weight_scale", (4, 2 * intermediate, 4)),
            ("down_proj_scale", "w2_weight_scale", (4, 128, intermediate // 32)),
        ]
    checkpoint, expected_shards = {}, {}
    for source_name, parameter_name, shape in specifications:
        values = torch.arange(math.prod(shape), device="cuda").remainder(97)
        values = values.to(torch.uint8 if quantized else torch.bfloat16).reshape(shape)
        checkpoint[parameter_name] = values
        model.load_weights(
            [(f"model.language_model.layers.0.mlp.experts.{source_name}", values)]
        )
        parameter = dict(moe.experts.named_parameters())[parameter_name]
        if parameter_name.startswith("w13"):
            gate, up = values.chunk(2, dim=1)
            expected = torch.cat(
                (
                    gate[
                        :, rank * local_intermediate : (rank + 1) * local_intermediate
                    ],
                    up[:, rank * local_intermediate : (rank + 1) * local_intermediate],
                ),
                dim=1,
            )
        else:
            columns = shape[-1] // ranks
            expected = values[:, :, rank * columns : (rank + 1) * columns]
        expected_shards[parameter_name] = expected.clone()
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)

    # Quark's separate expert projections must reload the same shards as the
    # fused checkpoint, including byte-packed columns and E8M0 block scales.
    updates = []
    for parameter_name, values in checkpoint.items():
        kind = "weight_scale" if parameter_name.endswith("_scale") else "weight"
        values = values + 1
        checkpoint[parameter_name] = values
        for expert in range(4):
            if parameter_name.startswith("w13"):
                gate, up = values[expert].chunk(2, dim=0)
                projections = (("gate_proj", gate), ("up_proj", up))
            else:
                projections = (("down_proj", values[expert]),)
            updates.extend(
                (
                    f"model.language_model.layers.0.mlp.experts.{expert}.{projection}.{kind}",
                    tensor,
                )
                for projection, tensor in projections
            )
    model.load_weights(updates)
    for parameter_name, expected in expected_shards.items():
        parameter = dict(moe.experts.named_parameters())[parameter_name]
        torch.testing.assert_close(parameter, expected + 1, atol=0, rtol=0)


@requires_gpu
def test_muse_spark_checkpoint_names_load_into_runtime_layout(
    bf16_default_dtype,
) -> None:
    model = _tiny_model()
    assert not any(
        name.startswith("model._final_norm_op") for name, _ in model.named_parameters()
    )
    expert_config = model.get_model_config_for_expert_location(model.config)
    assert (expert_config.num_layers, expert_config.num_logical_experts) == (1, 4)

    def full(shape: tuple[int, ...], value: float) -> torch.Tensor:
        return torch.full(shape, value, dtype=torch.bfloat16, device="cuda")

    # q and k vary along the row axis so the rotary pair restore is observable.
    q = (
        torch.arange(256, dtype=torch.bfloat16, device="cuda")
        .unsqueeze(1)
        .expand(256, 256)
        .contiguous()
    )
    k = (
        torch.arange(64, dtype=torch.bfloat16, device="cuda")
        .unsqueeze(1)
        .expand(64, 256)
        .contiguous()
    )
    v, o_gate = full((64, 256), 3.0), full((256, 256), 4.0)
    router = torch.arange(1024, dtype=torch.float32, device="cuda").view(4, 256)
    gate = torch.arange(256, dtype=torch.bfloat16, device="cuda")
    input_norm = torch.arange(256, dtype=torch.bfloat16, device="cuda") + 0.5
    w13 = torch.randn(4, 128, 128, dtype=torch.bfloat16, device="cuda")
    post_expert_norm = torch.arange(128, dtype=torch.bfloat16, device="cuda")
    vision_q, vision_k, vision_v = (
        full((32, 32), 4.0),
        full((32, 32), 5.0),
        full((32, 32), 6.0),
    )
    vision_projection = full((256, 64), 7.0)
    audio_conv = full((8, 1, 7), 8.0)
    model.load_weights(
        (
            ("model.language_model.layers.0.self_attn.q_proj.weight", q),
            ("model.language_model.layers.0.self_attn.k_proj.weight", k),
            ("model.language_model.layers.0.self_attn.v_proj.weight", v),
            ("model.language_model.layers.0.self_attn.gate_proj.weight", o_gate),
            ("model.language_model.layers.0.input_layernorm.weight", input_norm),
            ("model.language_model.layers.0.post_attention_residual_gate.gate", gate),
            ("model.language_model.layers.0.mlp.gate.weight", router),
            ("model.language_model.layers.0.mlp.experts.gate_up_proj", w13),
            (
                "model.language_model.layers.0.mlp.experts.post_expert_norm.weight",
                post_expert_norm,
            ),
            ("model.vision_tower.layers.0.attn.q_proj.weight", vision_q),
            ("model.vision_tower.layers.0.attn.k_proj.weight", vision_k),
            ("model.vision_tower.layers.0.attn.v_proj.weight", vision_v),
            ("model.vision_projection.weight", vision_projection),
            ("model.audio_tower.layers.0.conv_blocks.0.conv.weight", audio_conv),
        )
    )
    layer = model.model.layers[0]
    packed = layer.self_attn.qkv_proj.weight
    assert packed.shape[0] == 256 + 64 + 64 + 256
    assert torch.equal(
        packed,
        torch.cat(
            [restore_rotary_pairs(q, 32), restore_rotary_pairs(k, 32), v, o_gate]
        ),
    )
    assert not torch.equal(packed[:256], q)
    assert torch.equal(layer.input_layernorm.weight, input_norm)
    assert not hasattr(layer.self_attn, "gate_proj")
    assert torch.equal(layer.post_attention_residual_gate.gate, gate)
    # The router reads FP32 rows through a transposed weight.
    assert torch.equal(layer.mlp.gate.weight, router.T)
    assert torch.equal(layer.mlp.experts.w13_weight, w13)
    assert torch.equal(layer.mlp.post_expert_norm.weight, post_expert_norm)
    vision_qkv = model.vision_tower.layers[0].attn.qkv_proj.weight
    # Muse's rotary table is real-valued; the packed complex fast path must stay off.
    assert not model.vision_tower.layers[0].attn._use_packed_qkv_complex_rotary
    assert torch.equal(vision_qkv, torch.cat([vision_q, vision_k, vision_v]))
    assert torch.equal(model.vision_projection.weight, vision_projection)
    assert torch.equal(
        model.audio_tower.layers[0].conv_blocks[0].conv.weight, audio_conv
    )


def test_every_checkpoint_tensor_resolves_to_one_parameter_name() -> None:
    """Packed projections keep their shard; every other tensor maps one to one."""
    assert resolve_parameter_name("lm_head.weight") == ("lm_head.weight", None)
    assert resolve_parameter_name("model.language_model.embed_tokens.weight") == (
        "model.embed_tokens.weight",
        None,
    )
    assert resolve_parameter_name(
        "model.language_model.layers.7.self_attn.v_proj.weight"
    ) == ("model.layers.7.self_attn.qkv_proj.weight", "v")
    assert resolve_parameter_name(
        "model.language_model.layers.7.mlp.experts.down_proj_weight_scale_2"
    ) == ("model.layers.7.mlp.experts.w2_weight_scale_2", None)
    assert resolve_parameter_name(
        "model.language_model.layers.7.mlp.experts.post_expert_norm.weight"
    ) == ("model.layers.7.mlp.post_expert_norm.weight", None)
    assert resolve_parameter_name("model.vision_adapter.fc1.weight") == (
        "vision_tower.adapter.fc1.weight",
        None,
    )
    assert resolve_parameter_name("model.audio_tower.norm.weight") == (
        "audio_tower.norm.weight",
        None,
    )
    with pytest.raises(ValueError, match="no parameter"):
        resolve_parameter_name("model.language_model.layers.7.mlp.experts.up_proj")


def test_per_expert_tensors_resolve_to_their_fused_rows() -> None:
    """Quark's per-expert tensors fill one expert of the fused gate/up or down weight."""
    assert resolve_per_expert_tensor(
        "model.language_model.layers.7.mlp.experts.12.gate_proj.weight"
    ) == ("model.layers.7.mlp.experts.w13_weight", 12, 0)
    assert resolve_per_expert_tensor(
        "model.language_model.layers.7.mlp.experts.12.up_proj.weight_scale"
    ) == ("model.layers.7.mlp.experts.w13_weight_scale", 12, 1)
    assert resolve_per_expert_tensor(
        "model.language_model.layers.7.mlp.experts.3.down_proj.weight"
    ) == ("model.layers.7.mlp.experts.w2_weight", 3, None)
    assert (
        resolve_per_expert_tensor(
            "model.language_model.layers.7.mlp.experts.gate_up_proj"
        )
        is None
    )
    assert (
        resolve_per_expert_tensor(
            "model.language_model.layers.7.mlp.experts.post_expert_norm.weight"
        )
        is None
    )


def _quark_mxfp4_config(exclude: list[str]) -> Mxfp4Config:
    return Mxfp4Config.from_config(
        {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "scale_format": "e8m0",
                    "is_dynamic": False,
                },
                "input_tensors": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "scale_format": "e8m0",
                    "is_dynamic": True,
                },
            },
            "exclude": exclude,
        }
    )


def test_quark_per_projection_exclusions_exclude_the_fused_modules() -> None:
    attention = ("q_proj", "k_proj", "v_proj", "gate_proj")
    quant_config = _quark_mxfp4_config(
        [
            f"model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in range(2)
            for projection in attention
        ]
        + [
            f"model.language_model.layers.0.mlp.experts.{expert}.{projection}"
            for expert in range(2)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ]
    )
    exclude_fully_excluded_fused_modules(
        quant_config, num_layers=2, num_experts=2, prefix="model"
    )
    added = set(quant_config.ignored_layers)
    assert {
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.1.self_attn.qkv_proj",
        "model.layers.0.mlp.experts",
    } <= added
    assert "model.layers.1.mlp.experts" not in added


def test_quark_exclusions_reject_a_partially_excluded_fused_module() -> None:
    quant_config = _quark_mxfp4_config(
        ["model.language_model.layers.0.self_attn.q_proj"]
    )
    with pytest.raises(ValueError, match="only some projections"):
        exclude_fully_excluded_fused_modules(
            quant_config, num_layers=1, num_experts=2, prefix="model"
        )


@requires_gpu
def test_per_expert_checkpoint_tensors_load_into_the_fused_layout(
    bf16_default_dtype,
) -> None:
    model = _tiny_model()
    experts = model.model.layers[0].mlp.experts
    num_experts, fused_rows, hidden = experts.w13_weight.shape
    intermediate = fused_rows // 2
    torch.manual_seed(3)
    gates = torch.randn(num_experts, intermediate, hidden, device="cuda")
    ups = torch.randn(num_experts, intermediate, hidden, device="cuda")
    downs = torch.randn(num_experts, *experts.w2_weight.shape[1:], device="cuda")
    prefix = "model.language_model.layers.0.mlp.experts"
    model.load_weights(
        (name, tensor.to(torch.bfloat16))
        for expert in range(num_experts)
        for name, tensor in (
            (f"{prefix}.{expert}.gate_proj.weight", gates[expert]),
            (f"{prefix}.{expert}.up_proj.weight", ups[expert]),
            (f"{prefix}.{expert}.down_proj.weight", downs[expert]),
        )
    )
    assert torch.equal(
        experts.w13_weight, torch.cat([gates, ups], dim=1).to(torch.bfloat16)
    )
    assert torch.equal(experts.w2_weight, downs.to(torch.bfloat16))


@requires_gpu
def test_vision_tower_encodes_flat_image_patches_to_text_width(
    bf16_default_dtype,
) -> None:
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.randn(4, 24, device="cuda"),
        model_specific_data={
            "image_grid_thw": torch.tensor([[1, 2, 2]], device="cuda")
        },
    )
    model = _tiny_model()
    # The parameters are allocated uninitialized; give them finite values.
    torch.manual_seed(0)
    for parameter in model.parameters():
        parameter.data.normal_(0.0, 0.02)
    output = model.get_vision_feature([item])
    assert output.shape == (1, 256) and output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_modelopt_substring_exclusions_widen_to_matching_globs() -> None:
    quant_config = SimpleNamespace(
        exclude_modules=[
            "self_attn",
            "lm_head",
            "layers.0.",
            "layers.61.",
            "*already.globbed*",
        ]
    )
    _widen_modelopt_exclusions(quant_config)
    excluded = list(quant_config.exclude_modules)
    assert excluded == [
        "*self_attn*",
        "*lm_head*",
        "*layers.0.*",
        "*layers.61.*",
        "*already.globbed*",
    ]
    for prefix in (
        "model.layers.0.mlp.experts",
        "model.layers.61.mlp.experts",
        "model.layers.5.self_attn.qkv_proj",
        "lm_head",
    ):
        assert should_exclude_quant_module(prefix, excluded)
    for prefix in (
        "model.layers.5.mlp.experts",
        "model.layers.10.mlp.experts",
    ):
        assert not should_exclude_quant_module(prefix, excluded)
    # A second pass must not re-wrap; absent or empty lists are left alone.
    _widen_modelopt_exclusions(quant_config)
    assert quant_config.exclude_modules == excluded
    _widen_modelopt_exclusions(None)
    empty = SimpleNamespace(exclude_modules=[])
    _widen_modelopt_exclusions(empty)
    assert empty.exclude_modules == []


def test_vision_tower_permutations_preserve_patch_geometry() -> None:
    device = torch.device("cpu")
    order, offset = pixel_shuffle_permutation(
        offset=0, grid_h=4, grid_w=4, factor=2, device=device
    )
    assert offset == 16
    assert order.tolist() == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    shuffled = downsample_pixel_shuffle(
        value=torch.arange(16).view(16, 1), permutation=order, factor=2
    )
    assert shuffled.tolist() == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]
    sparse, lengths, sparse_offset = sparse_permutation(
        offset=0, grid_h=3, grid_w=5, block_h=2, block_w=3, device=device
    )
    assert sparse_offset == 15
    assert sorted(sparse.tolist()) == list(range(15))
    assert lengths == [6, 4, 3, 2]


@requires_gpu
def test_embedding_norm_applies_to_supplied_input_embeds() -> None:
    """The prefill graph lands raw token embeddings, so they must be normalized."""
    model = _stub(MuseSparkForConditionalGeneration.model_cls)
    model.config = SimpleNamespace(rms_norm_eps=1e-5)
    torch.manual_seed(3)
    raw = torch.randn(2, 8, device="cuda") * 5.0
    hidden = model._embed_hidden(input_ids=None, input_embeds=raw)
    torch.testing.assert_close(hidden, _rmsnorm(raw.float(), 1e-5))
    rms = hidden.square().mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)
    normalized = _rmsnorm(torch.randn(4, 8, device="cuda"), 1e-5)
    once = model._embed_hidden(input_ids=None, input_embeds=normalized)
    torch.testing.assert_close(once, normalized.float(), atol=1e-3, rtol=1e-3)


def test_attention_leaves_norm_rope_and_kv_write_to_the_prologue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The layer hands raw q/k/v and positions to PagedAttention, whose
    prologue normalizes, rotates and writes K/V."""
    heads, kv_heads, head_dim = 4, 2, 8
    attention = _stub(MuseSparkAttention)
    attention.hidden_size, attention.head_dim = heads * head_dim, head_dim
    attention.num_heads, attention.num_kv_heads = heads, kv_heads
    attention.q_size, attention.kv_size = heads * head_dim, kv_heads * head_dim
    attention.rms_norm_eps = 1e-5
    width = 2 * attention.q_size + 2 * attention.kv_size
    rows = torch.arange(3 * width, dtype=torch.bfloat16).view(3, width)
    attention.qkv_proj = lambda x: (rows, None)
    attention._project = lambda x: (x, None, 1)
    attention.attn = Mock(return_value=torch.zeros(3, attention.q_size))
    monkeypatch.setattr(
        module, "weightless_rmsnorm_gated_sigmoid", lambda out, *args: out
    )
    monkeypatch.setattr(module, "pdl_enabled", lambda: False)
    positions, ctx = torch.arange(3), SimpleNamespace()
    attention(positions, torch.zeros(3, attention.hidden_size), ctx, None)
    q, k, v, positions_arg, ctx_arg = attention.attn.call_args.args
    q_end, kv_end = attention.q_size, attention.q_size + attention.kv_size
    assert torch.equal(q, rows[:, :q_end])
    assert torch.equal(k, rows[:, q_end:kv_end])
    assert torch.equal(v, rows[:, kv_end : kv_end + attention.kv_size])
    assert positions_arg is positions and ctx_arg is ctx


@requires_gpu
@pytest.mark.parametrize("with_rope", [False, True])
@pytest.mark.parametrize("world_size", [4, 8])
def test_owned_attention_stores_prepared_rows_once(
    monkeypatch: pytest.MonkeyPatch, with_rope: bool, world_size: int
) -> None:
    """Prepared receiver outputs pass through the real prologue's cache store,
    preserving prefix slots and reaching attention without another head norm."""
    from tokenspeed_kernel.ops.attention.prologue import HeadKVCache

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers import paged_attention

    rows, head_dim, heads, kv_heads = 2049, 64, 128 // world_size, 16 // world_size
    attention = _stub(MuseSparkAttention)
    attention.tp_size = world_size
    attention.hidden_size, attention.head_dim = 8192, head_dim
    attention.num_heads, attention.num_kv_heads = heads, kv_heads
    attention.q_size, attention.kv_size = heads * head_dim, kv_heads * head_dim
    attention.rms_norm_eps, attention.q_scale = 1e-5, 5.4730064863875
    attention.q_norm = module._ScaledHeadNorm(
        head_dim, attention.q_scale, attention.rms_norm_eps
    ).cuda()
    attention.k_norm = module._ScaledHeadNorm(
        head_dim, 1.0, attention.rms_norm_eps
    ).cuda()
    positions = torch.arange(rows, device="cuda") + 3
    table = torch.randn(rows + 3, head_dim, device="cuda")
    attention.rotary_emb = (
        SimpleNamespace(
            cos_sin_cache=table,
            as_rotary=lambda _: pytest.fail("prepared rows rotated a second time"),
        )
        if with_rope
        else None
    )
    attention.nope_cos_sin_cache = torch.empty(1, 0, device="cuda")
    attention.attn = paged_attention.PagedAttention(
        heads,
        head_dim,
        head_dim**-0.5,
        num_kv_heads=kv_heads,
        layer_id=0,
        rotary_emb=attention.rotary_emb,
        qk_norm=(attention.q_norm, attention.k_norm),
    )
    prepared_q, prepared_k, prepared_v, gate = (
        torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
        for width in (
            attention.q_size,
            attention.kv_size,
            attention.kv_size,
            attention.q_size,
        )
    )
    local_rows = (rows + world_size - 1) // world_size
    local = torch.empty(local_rows, 8192, dtype=torch.bfloat16, device="cuda")
    attention.qkv_proj = SimpleNamespace(
        project_owned_rows=lambda value, global_rows: value,
        tp_group=tuple(range(world_size)),
    )
    attention._project = lambda value: (value, None, 1)

    attention.iris_state = SimpleNamespace(
        all_to_all_transpose_qkvg_norm_rope=lambda *args: (
            prepared_q,
            prepared_k,
            prepared_v,
            gate,
        )
    )
    monkeypatch.setattr(
        module, "weightless_rmsnorm_gated_sigmoid", lambda value, *args: value
    )
    monkeypatch.setattr(module, "pdl_enabled", lambda: False)
    k_cache = torch.full(
        (rows + 3, kv_heads, head_dim), 17, dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.full_like(k_cache, -19)
    write_targets = []

    def write_target(layer_id, slots, mask):
        write_targets.append(slots)
        return HeadKVCache(k_cache, v_cache, None, slots)

    def attend(q, k, v, layer, pool, mode, bs, **kwargs):
        assert kwargs["save_kv_cache"] is False
        assert torch.equal(k.flatten(1), prepared_k)
        assert torch.equal(v.flatten(1), prepared_v)
        return q

    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        bs=1,
        attn_backend=SimpleNamespace(
            padded_write_locations=lambda *args: positions,
            cache_placement=lambda layer: None,
            forward=attend,
        ),
        token_to_kv_pool=SimpleNamespace(
            dtype=torch.bfloat16, kv_write_target=write_target
        ),
    )
    output, sentinel, ranks = attention(positions, local, ctx, rows)
    assert sentinel is None and ranks == 1
    assert torch.equal(output, prepared_q)
    assert len(write_targets) == 1
    assert torch.equal(k_cache[3:].flatten(1), prepared_k)
    assert torch.equal(v_cache[3:].flatten(1), prepared_v)
    assert torch.all(k_cache[:3] == 17) and torch.all(v_cache[:3] == -19)


def test_query_norm_keeps_the_checkpoint_scale_unrounded() -> None:
    """The prologue forms ``weight + offset`` in fp32, so an fp32 weight keeps
    the query scale the fused kernel applied before the port."""
    scale = 5.4730064863875
    q_norm = module._ScaledHeadNorm(8, scale, 1e-5)
    k_norm = module._ScaledHeadNorm(8, 1.0, 1e-5)
    assert q_norm.weight.dtype == torch.float32 and q_norm.weight_offset == 0.0
    assert torch.equal(q_norm.weight, torch.full((8,), scale, dtype=torch.float32))
    assert torch.equal(k_norm.weight, torch.ones(8))
    assert q_norm.variance_epsilon == k_norm.variance_epsilon == 1e-5


def test_mixed_rounds_are_rejected_before_any_attention_runs() -> None:
    ctx = SimpleNamespace(forward_mode=SimpleNamespace(is_mixed=lambda: True))
    with pytest.raises(ValueError, match="mixed"):
        MuseSparkDecoderLayer.forward(
            SimpleNamespace(), torch.arange(3), torch.zeros(3, 8), ctx, None, None, None
        )


def test_packed_qkv_gate_loader_replicates_kv_heads_across_ranks() -> None:
    """Q and the gate shard by rank; K/V follow the replicated KV head."""
    hidden, head_dim, heads, kv_heads = 32, 8, 4, 1
    q_width, kv_width = heads * head_dim, kv_heads * head_dim
    full = {
        name: torch.arange(rows * hidden, dtype=torch.float32).view(rows, hidden)
        + 1000 * i
        for i, (name, rows) in enumerate(
            (("q", q_width), ("k", kv_width), ("v", kv_width), ("g", q_width))
        )
    }
    for rank in (0, 1):
        layer = QKVOGateParallelLinear(
            hidden_size=hidden,
            head_size=head_dim,
            total_num_heads=heads,
            total_num_kv_heads=kv_heads,
            quant_config=None,
            prefix="qkv_proj",
            tp_rank=rank,
            tp_size=2,
            tp_group=(0, 1),
            replicate_for_owned_rows=False,
        )
        assert layer.num_kv_head_replicas == 2 and layer.num_kv_heads == 1
        for shard, weight in full.items():
            layer.weight_loader(layer.weight, weight, shard)
        q_rows = slice(rank * q_width // 2, (rank + 1) * q_width // 2)
        expected = torch.cat(
            [full["q"][q_rows], full["k"], full["v"], full["g"][q_rows]]
        )
        assert torch.equal(layer.weight.data, expected)


def _forward_ctx(decode: bool) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: decode)
    )


def _mm_context(extend: bool) -> SimpleNamespace:
    return SimpleNamespace(has_extend_inputs=lambda: extend)


def _seam_model(monkeypatch: pytest.MonkeyPatch, model_kwargs: dict):
    """A stub multimodal model whose embedder hands back a marker tensor."""
    model = _stub(MuseSparkForConditionalGeneration)
    model.is_multimodal_active = True
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 4)
    merged = torch.full((3, 4), 7.0)
    seen: dict = {}
    model.multimodal_embedder = SimpleNamespace(
        apply=lambda **kwargs: seen.update(kwargs) or (merged, model_kwargs)
    )
    monkeypatch.setattr(model, "get_multimodal_encoder_specs", lambda: {"spec": 1})
    return model, merged, seen


def test_multimodal_input_embeds_merges_only_new_prefill_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefill graph lands what this seam returns; anything else fails loudly."""
    model_kwargs: dict = {}
    model, merged, seen = _seam_model(monkeypatch, model_kwargs)
    ids = torch.arange(3)
    assert model.multimodal_input_embeds(ids, _forward_ctx(False), None) is None
    assert (
        model.multimodal_input_embeds(ids, _forward_ctx(False), _mm_context(False))
        is None
    )
    assert (
        model.multimodal_input_embeds(ids, _forward_ctx(True), _mm_context(True))
        is None
    )
    assert seen == {}
    assert (
        model.multimodal_input_embeds(ids, _forward_ctx(False), _mm_context(True))
        is merged
    )
    assert (
        seen["input_ids"] is ids and seen["text_embedding"] is model.model.embed_tokens
    )
    assert seen["encoders"] == {"spec": 1} and seen["multimodal_model"] is model
    model.is_multimodal_active = False
    with pytest.raises(RuntimeError, match="encoders are disabled"):
        model.multimodal_input_embeds(ids, _forward_ctx(False), _mm_context(True))


def test_forward_hands_the_merged_embeddings_to_the_shared_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, merged, _ = _seam_model(monkeypatch, {})
    calls: list[dict] = []
    monkeypatch.setattr(
        module.BaseCausalLM,
        "forward",
        lambda self, ctx, input_ids, positions, **kwargs: calls.append(kwargs) or "out",
    )
    ids = torch.arange(3)
    assert (
        model.forward(
            _forward_ctx(False), ids, ids, multimodal_context=_mm_context(True)
        )
        == "out"
    )
    assert calls[-1] == {"input_embeds": merged}
    assert (
        model.forward(_forward_ctx(False), ids, ids, multimodal_context=None) == "out"
    )
    assert calls[-1] == {}


def test_a_gated_projection_needs_one_activation_scale_for_both_halves() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    halves = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="one set of rows"):
        module.load_expert_input_scale(parameter, halves, 0)
    module.load_expert_input_scale(
        parameter,
        torch.tensor([[1.0, 1.0], [3.0, 3.0]]),
        0,
    )
    assert parameter.item() == 3.0


def test_a_per_expert_scale_keeps_the_rank_its_own_experts() -> None:
    """The second GEMM reads each expert's own rows, so every rank keeps its own slice."""
    parameter = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
    module.load_expert_input_scale(parameter, scales, 2)
    assert parameter.tolist() == [3.0, 4.0]
    module.load_expert_input_scale(parameter, scales, 0)
    assert parameter.tolist() == [1.0, 2.0]


@requires_gpu
def test_reloaded_gates_refresh_the_cached_coefficients_in_place() -> None:
    """A weight update rewrites the gate parameter; alpha/beta must follow it."""
    layer = _stub(MuseSparkDecoderLayer)
    layer.residual_gate_temperature = 0.3
    layer.post_attention_residual_gate = module.MuseSparkResidualGate(64, 0.3).cuda()
    layer.post_feedforward_residual_gate = module.MuseSparkResidualGate(64, 0.3).cuda()
    layer.post_attention_residual_gate.gate.data.copy_(
        torch.arange(64, dtype=torch.bfloat16, device="cuda") - 32
    )
    layer.post_feedforward_residual_gate.gate.data.zero_()
    layer.refresh_residual_gates()
    first = layer.post_attention_residual_gate._gate_coefficients()
    attn_gate = layer.post_attention_residual_gate.gate
    attn_gate.data.copy_(-attn_gate.data)
    layer.refresh_residual_gates()
    second = layer.post_attention_residual_gate._gate_coefficients()
    assert second[0].data_ptr() == first[0].data_ptr()
    expected = residual_gate_coefficients(attn_gate, 0.3, RESIDUAL_GATE_EPS)
    assert torch.equal(second[0], expected[0]) and torch.equal(second[1], expected[1])


def test_quantized_combined_prepares_routing_once_per_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The experts get the prepared routing when it exists; otherwise one is prepared for next time."""
    moe = _stub(MuseSparkMoE)
    moe.expert_quant_kind = "nvfp4"
    moe.iris_state = None
    moe.tp_size, moe.ep_size = 1, 1
    moe.topk, moe.prepared_routing = 2, {}
    moe.ep_group = None
    moe.post_expert_norm = SimpleNamespace(
        weight=torch.ones(8), eps=1e-5, weight_offset=0.0
    )
    moe._reduce_expert_routes = lambda **kwargs: kwargs["routes"]
    seen: list[object] = []

    def experts(**kwargs):
        seen.append(kwargs["routing_prepared"])
        return torch.zeros(4, 8), None, torch.arange(4, dtype=torch.int32)

    prepared_for: list[int] = []

    def prepare(experts_module, x_quantized, topk_ids, topk_weights, activation, pdl):
        assert experts_module is moe.experts and activation is None
        prepared_for.append(topk_ids.shape[0])
        return SimpleNamespace(num_tokens=topk_ids.shape[0])

    moe.experts = experts
    monkeypatch.setattr(module, "prepare_nvfp4_routing", prepare)
    monkeypatch.setattr(module, "pdl_enabled", lambda: False)

    def combine(tokens: int, prepared: object) -> None:
        quantized = (
            torch.zeros(tokens, 4, dtype=torch.uint8),
            torch.zeros(tokens, 1, dtype=torch.uint8),
        )
        weights, ids = torch.ones(tokens, 2), torch.zeros(tokens, 2, dtype=torch.int32)
        moe._quantized_combined(
            torch.zeros(tokens, 8),
            quantized,
            weights,
            ids,
            None,
            tokens,
            tokens,
            prepared,
            False,
        )

    combine(2, None)
    assert seen == [None] and prepared_for == [2]
    handle = moe.prepared_routing[2]
    combine(2, handle)
    assert seen[-1] is handle and prepared_for == [2]
    # A second miss at the same count and any count past the ceiling prepare nothing.
    combine(2, None)
    combine(module.FUSED_ROUTING_MAX_TOKENS + 1, None)
    assert prepared_for == [2] and list(moe.prepared_routing) == [2]

    moe.expert_quant_kind = "mxfp4"
    combine(3, None)
    assert prepared_for == [2] and list(moe.prepared_routing) == [2]


def test_select_experts_routes_in_the_topk_kernel_when_prepared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared GEMMs take the fused top-k + routing launch; otherwise the plain top-k runs."""
    moe = _stub(MuseSparkMoE)
    moe.topk, moe.route_scale, moe.expert_start, moe.experts_per_rank = 2, 1.5, 4, 4
    moe.gate = SimpleNamespace(e_score_correction_bias=torch.zeros(8))
    logits = torch.randn(3, 8)
    calls: list[str] = []
    monkeypatch.setattr(module, "pdl_enabled", lambda: False)
    monkeypatch.setattr(
        module,
        "moe_topk",
        lambda *args, **kwargs: calls.append("topk") or ("w", "i"),
    )

    def fused(*args):
        calls.append("fused")
        assert args[:2] == (logits, moe.gate.e_score_correction_bias)
        assert args[2:8] == (2, 1.5, True, 4, 4, routing)
        return "fw", "fi"

    monkeypatch.setattr(module, "moe_sigmoid_bias_topk_routing", fused)
    routing = object()
    assert moe._select_experts(logits, None) == ("w", "i")
    assert moe._select_experts(logits, SimpleNamespace(routing=routing)) == ("fw", "fi")
    assert calls == ["topk", "fused"]


def test_restore_rotary_pairs_interleaves_each_head() -> None:
    head_dim = 8
    rows = torch.arange(2 * head_dim, dtype=torch.float32).unsqueeze(1)
    restored = restore_rotary_pairs(rows, head_dim)
    assert restored.squeeze(1).tolist() == [
        0.0,
        4.0,
        1.0,
        5.0,
        2.0,
        6.0,
        3.0,
        7.0,
        8.0,
        12.0,
        9.0,
        13.0,
        10.0,
        14.0,
        11.0,
        15.0,
    ]


def test_vision_tower_rope_rotates_the_two_halves_of_each_head() -> None:
    """The vision tower pairs a head's dimension i with i + head_dim // 2, not with i + 1."""
    head_dim = 4
    query = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    key = torch.zeros_like(query)
    angle = math.pi / 2
    frequencies = torch.tensor(
        [[[math.cos(angle), math.sin(angle)], [math.cos(angle), math.sin(angle)]]]
    )
    rotated, _ = apply_complex_rope(query, key, frequencies, (1, 1, head_dim))
    # cos is 0 and sin is 1, so the halves swap with a sign on the first.
    torch.testing.assert_close(
        rotated, torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]]), atol=1e-6, rtol=1e-6
    )


def _shard_linear(
    out_features: int,
    in_features: int,
    tp_rank: int,
    tp_size: int,
) -> module._MuseGatheredLinear:
    linear = object.__new__(module._MuseGatheredLinear)
    ColumnParallelLinear.__init__(
        linear,
        input_size=in_features,
        output_size=out_features,
        bias=False,
        gather_output=False,
        skip_bias_add=False,
        params_dtype=None,
        quant_config=None,
        output_sizes=None,
        prefix="pre_expert_proj",
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_group=tuple(range(tp_size)) if tp_size > 1 else None,
        use_presharded_weights=False,
        override_kernel_name=None,
        interleave_linear_and_gate=False,
    )
    linear.rank = tp_rank
    linear.group = tuple(range(tp_size)) if tp_size > 1 else (0,)
    linear.workspace = None
    linear.multicast = None
    linear.gather_column_view = False
    return linear


@requires_gpu
@pytest.mark.parametrize("quantized", [False, True])
def test_attention_output_preserves_dense_projection(
    monkeypatch: pytest.MonkeyPatch,
    quantized: bool,
) -> None:
    tokens = 3
    if quantized and not current_platform().is_cdna4:
        pytest.skip("dense dynamic MXFP4 requires CDNA4")
    config = (
        Mxfp4Config(
            is_checkpoint_mxfp4_serialized=True,
            use_dynamic_mxfp4_activations=True,
        )
        if quantized
        else None
    )
    with torch.device("cuda"):
        linear = RowParallelLinear(
            input_size=1024,
            output_size=256,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="model.layers.0.self_attn.o_proj",
            tp_rank=0,
            tp_size=4,
            tp_group=(0, 1, 2, 3),
        )
    if quantized:
        linear.weight.data.fill_(0x22)
        linear.weight_scale.data.fill_(126)
        weight = torch.full((256, 256), 0.5, device="cuda", dtype=torch.bfloat16)
    else:
        torch.manual_seed(7)
        linear.weight.data.normal_(std=0.02)
        weight = linear.weight
    row = (torch.arange(256, device="cuda") % 8 - 4).to(torch.bfloat16) / 2
    x = row.expand(tokens, -1).contiguous()
    output = torch.full((tokens, 256), torch.nan, device="cuda", dtype=x.dtype)
    monkeypatch.setattr(
        module,
        "_acquire_all_reduce_output",
        lambda shape, like, group: output,
    )
    attention = _stub(MuseSparkAttention)
    attention.o_proj = linear
    result = attention._project_output(x)

    torch.testing.assert_close(result, x @ weight.t(), atol=1e-2, rtol=1e-2)
    if quantized:
        assert output.isnan().all()
    else:
        assert result.data_ptr() == output.data_ptr()


@requires_gpu
@pytest.mark.parametrize("tokens", [2, 16])
def test_gathered_projection_writes_decode_rows_into_the_caller_buffer(
    bf16_default_dtype, tokens: int
) -> None:
    """Multi-row decode batches land in the gather input, not a copy of a new result."""
    from tokenspeed_kernel.ops.gemm import mm

    if not current_platform().is_cdna4:
        pytest.skip("the direct gather-slice projection is the CDNA4 path")
    torch.manual_seed(5)
    with torch.device("cuda"):
        linear = _shard_linear(512, 8192, tp_rank=0, tp_size=1)
    linear.weight.data.normal_(std=0.02)
    x = torch.randn((tokens, 8192), device="cuda", dtype=torch.bfloat16)
    local = torch.full((tokens, 512), torch.nan, device="cuda")
    linear._project(x, linear.weight, local)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    result = linear._project(x, linear.weight, local)

    # Copying through a fresh [tokens, 512] result would raise the peak.
    assert torch.cuda.max_memory_allocated() == before
    assert result is local
    assert torch.equal(local, mm(x, linear.weight))


def _gathered_mxfp4_workspace_worker(rank: int, port: int, world_size: int) -> None:
    from tokenspeed_kernel.ops.communication._iris.muse_spark import (
        prepare_muse_communication,
    )
    from tokenspeed_kernel.ops.quantization import rmsnorm_mxfp4_quant

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cuda", rank)
        state = prepare_muse_communication(
            enable_lamport=False,
            owned_route_max_rows=0,
            all_reduce=None,
            group=dist.group.WORLD,
            rank_in_group=rank,
            staged_max_numel=128 * 4096,
            producer_direct_max_numel=128 * 4096,
            device=device,
        )
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            linear = _shard_linear(4096, 256, tp_rank=rank, tp_size=world_size)
        linear.weight.data.zero_()
        linear.weight.data[:, rank] = 1
        linear.workspace = torch.zeros(64, 4096, device=device)
        linear.iris_state = state
        norm_weight = torch.ones(4096, device=device)
        # Route the layer through the real peer kernels without a serving process.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                module,
                "all_gather",
                lambda local, group, dim, backend: state.gather.run(local),
            )
            for tokens in (64, 65):
                x = torch.ones(tokens, 256, device=device)
                x[:, :world_size] = torch.arange(1, world_size + 1, device=device)
                rows, packed = linear.forward_mxfp4(x, norm_weight, 1e-6, 1.0)
                assert rows.shape == (tokens, 4096)
                expected_rows = x[:, :world_size].repeat_interleave(
                    4096 // world_size, dim=1
                )
                expected = rmsnorm_mxfp4_quant(expected_rows, norm_weight, 1e-6, 1.0)
                torch.testing.assert_close(packed[0], expected[0], atol=0, rtol=0)
                m = torch.arange(tokens, device=device)[:, None]
                k = torch.arange(128, device=device)[None, :]
                offsets = (
                    ((((k // 8 * 4 + k % 4) * 16 + m % 16) * 2 + k % 8 // 4) * 2)
                    + m % 32 // 16
                    + m // 32 * 4096
                )
                torch.testing.assert_close(
                    packed[1].T.flatten()[offsets],
                    expected[1].T.flatten()[offsets],
                    atol=0,
                    rtol=0,
                )
                if tokens > 64:
                    torch.testing.assert_close(rows, expected_rows, atol=0, rtol=0)
                torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


@requires_gpu
@pytest.mark.parametrize("world_size", [4, 8])
def test_gathered_mxfp4_preserves_rows_across_workspace_boundary(world_size) -> None:
    if not current_platform().is_cdna4 or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CDNA4 GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _gathered_mxfp4_workspace_worker,
        args=(port, world_size),
        nprocs=world_size,
        join=True,
    )


@requires_gpu
@pytest.mark.parametrize("world_size", [4, 8])
def test_replicated_pool_preserves_column_projection_after_reload(
    bf16_default_dtype, world_size
):

    torch.manual_seed(123)
    full_weight = torch.randn((4096, 8192), device="cuda", dtype=torch.bfloat16)
    inputs = [
        torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
        for rows in (1, 17)
    ]
    for rank in range(world_size):
        pre_expert = object.__new__(module._MuseGatheredLinear)
        ColumnParallelLinear.__init__(
            pre_expert,
            8192,
            4096,
            bias=False,
            params_dtype=torch.bfloat16,
            tp_rank=0,
            tp_size=1,
            tp_group=None,
        )
        pre_expert = pre_expert.cuda()
        pre_expert.rank, pre_expert.group = rank, tuple(range(world_size))
        pre_expert.gather_column_view = True
        pre_expert.workspace = None
        pre_expert.multicast = None
        reference = _shard_linear(4096, 8192, rank, world_size).cuda()
        for scale in (1.0, -0.5):
            loaded = full_weight * scale
            pre_expert.weight_loader(pre_expert.weight, loaded)
            reference.weight_loader(reference.weight, loaded)
            assert pre_expert._local_weight().is_contiguous()
            for x in inputs:
                torch.testing.assert_close(pre_expert(x), reference(x), atol=0, rtol=0)
            owned = inputs[1][:5].contiguous()
            output = torch.empty((5, 4096), device="cuda", dtype=torch.bfloat16)
            actual = pre_expert.forward_owned_rows(owned, output)
            torch.testing.assert_close(
                actual, torch.mm(owned, loaded.T), atol=0, rtol=0
            )
            assert actual.data_ptr() == output.data_ptr()


@requires_gpu
@pytest.mark.skipif(
    not current_platform().is_nvidia, reason="requires a NVIDIA CUDA pointer view"
)
def test_gathered_projection_hands_wider_batches_to_the_mailbox_publish(
    monkeypatch: pytest.MonkeyPatch, bf16_default_dtype
) -> None:
    """Wider rows reach the mailbox through its word publish, bit for bit; a GEMV row lands in place."""
    from tokenspeed_kernel.ops.moe._cute_dsl import latent_down

    bound = {}
    monkeypatch.setattr(
        module.process_group_manager, "get_device_process_group", lambda group: group
    )
    monkeypatch.setattr(
        module.KimiK3LatentDownOp, "initialize", lambda **kwargs: bound.update(kwargs)
    )
    hidden, latent, ranks, rank = 256, 128, 4, 1
    with torch.device("cuda"):
        linear = module._MuseGatheredLinear(
            input_size=hidden,
            output_size=latent,
            prefix="model.layers.0.mlp.pre_expert_proj",
            tp_rank=rank,
            tp_size=ranks,
            tp_group=tuple(range(ranks)),
            layer_id=0,
            layer_count=2,
            replicate_for_owned_rows=True,
        )
    with torch.no_grad():
        linear.weight.normal_(std=hidden**-0.5)
    weight = linear._local_weight()
    shard = weight.shape[0]
    mailbox = torch.zeros(4, latent, dtype=torch.bfloat16, device="cuda")
    latent_down.arm_mailbox(mailbox, latent_down.HALF_POLLED_SENTINEL)
    producer = latent_down._MulticastVaGemm(
        multicast_ptr=mailbox.data_ptr(),
        rank=rank,
        shard_dim=shard,
        latent_size=latent,
        max_m=4,
        device=mailbox.device,
        publish_gemm=bound["publish_gemm"],
    )
    published_rows, expected_rows = [], []
    publish = latent_down.multicast_publish

    def record_publish(block, destination):
        published_rows.append(block.shape[0])
        publish(block, destination)

    monkeypatch.setattr(latent_down, "multicast_publish", record_publish)
    for tokens in (1, 3):
        x = torch.randn(tokens, hidden, device="cuda")
        if tokens > 1 or not module._use_mv(x, weight):
            expected_rows.append(tokens)
        producer(x, weight, mailbox, mailbox.data_ptr())
        torch.cuda.synchronize()
        assert torch.equal(
            mailbox[:tokens, rank * shard : (rank + 1) * shard].view(torch.int16),
            linear._project(x, weight, None).view(torch.int16),
        )
    assert published_rows == expected_rows


class _FakeMailbox:
    sentinel = 0xFFFFFFFF

    def __init__(self, calls: list[tuple[str, int, int]]) -> None:
        self.calls = calls

    def handles(self, tokens: int) -> bool:
        return tokens <= 2

    def publish(self, x, weight):
        self.calls.append(("publish", x.shape[0], weight.shape[0]))
        return x.new_full((x.shape[0], 512), float("nan"))

    def __call__(self, x, weight, *, output_scale):
        # The real op's signature: Muse gathers BF16, never fused NVFP4.
        assert output_scale is None
        self.calls.append(("gather", x.shape[0], weight.shape[0]))
        return x.new_zeros(x.shape[0], 512)


@requires_gpu
def test_attention_publishes_a_served_batch_for_the_boundary_to_reduce(
    monkeypatch: pytest.MonkeyPatch, bf16_default_dtype
) -> None:
    """A served decode batch is published with its sentinel and rank count; the rest is projected."""
    attn = _tiny_model().model.layers[0].self_attn
    # A single rank has nothing to reduce.
    assert attn.o_proj_multicast is None
    with torch.no_grad():
        attn.o_proj.weight.normal_(std=attn.o_proj.weight.shape[1] ** -0.5)
    calls: list[tuple[str, int, int]] = []
    attn.o_proj_multicast = _FakeMailbox(calls)
    attn.tp_size = 4
    monkeypatch.setattr(
        module, "boundary_gathers_mailbox", lambda rows, width, ranks: ranks == 4
    )
    monkeypatch.setattr(
        module, "current_platform", lambda: SimpleNamespace(is_cdna4=False)
    )
    width = attn.o_proj.weight.shape[1]
    one = torch.randn(1, width, dtype=torch.bfloat16, device="cuda")
    rows, sentinel, ranks = attn._project(one)
    assert (sentinel, ranks) == (0xFFFFFFFF, 4) and rows.isnan().all()
    three = torch.randn(3, width, dtype=torch.bfloat16, device="cuda")
    rows, sentinel, ranks = attn._project(three)
    assert (sentinel, ranks) == (None, 1)
    assert rows.shape == (3, attn.hidden_size) and not rows.isnan().any()
    assert calls == [("publish", 1, attn.o_proj.weight.shape[0])]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
