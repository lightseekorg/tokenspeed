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

"""Weight loading tests for Hy4-flash FP8 128x128, BF16 scale_inv, ViT, and DFly2."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.configs.hy4_flash_config import (  # noqa: E402
    HYV4FlashConfig,
    HYV4FlashDFlyConfig,
    HYV4FlashTextConfig,
    HYV4FlashVisionConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config  # noqa: E402
from tokenspeed.runtime.models.hy4_flash import (  # noqa: E402
    HYV4FlashForConditionalGeneration,
)
from tokenspeed.runtime.models.hy4_flash_dfly import (  # noqa: E402
    HYV4FlashForCausalLMDFly,
)


def test_multimodal_fp8_checkpoint_loading_and_prefix_rewriting():
    quant_dict = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "language_model.model.lm_head",
            "language_model.model.model.embed_tokens",
            "language_model.model.model.hc_head.hc_head_fn",
            "language_model.model.model.layers.0.mlp.down_proj",
            "language_model.model.model.layers.0.mlp.gate_proj",
            "language_model.model.model.layers.0.mlp.up_proj",
            "language_model.model.model.layers.0.self_attn.kv_a_proj_with_mqa",
            "language_model.model.model.layers.0.self_attn.kv_b_proj",
            "language_model.model.model.layers.0.self_attn.linear_gate",
            "language_model.model.model.layers.0.self_attn.o_proj",
            "language_model.model.model.layers.0.self_attn.q_a_proj",
            "language_model.model.model.layers.0.self_attn.q_b_proj",
            "language_model.model.model.layers.1.self_attn.q_a_proj",
            "language_model.model.model.layers.1.self_attn.kv_a_proj_with_mqa",
            "language_model.model.model.layers.1.self_attn.kv_b_proj",
            "language_model.model.model.layers.1.mlp.gate",
            "language_model.model.model.layers.1.mlp.shared_experts.down_proj",
            "language_model.model.model.layers.1.mlp.shared_experts.gate_proj",
            "language_model.model.model.layers.1.mlp.shared_experts.up_proj",
            "language_model.model.model.layers.1.mlp.routed_expert_down_proj",
            "language_model.model.model.layers.1.mlp.routed_expert_up_proj",
            "language_model.model.model.layers.1.engram_layer.value_proj",
            "vit",
        ],
    }
    text_cfg = HYV4FlashTextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        routed_expert_hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        swa_num_attention_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=8,
        index_kpool=4,
        sliding_window=16,
        layer_types=["sliding_attention", "sliding_attention"],
        mlp_layer_types=["dense", "sparse"],
        hc_mult=4,
        engram_layer_ids=[2],
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=4,
        engram_n_embed_per_ngram=64,
        engram_layer_head_vocab_sizes={"2": [[31, 37, 41, 43], [47, 53, 59, 61]]},
        engram_pad_id=0,
        engram_lookup_table_size=256,
        enable_lm_head_fp32=True,
        quantization_config=quant_dict,
    )
    vit_cfg = HYV4FlashVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        patch_size=16,
        spatial_merge_size=2,
        out_hidden_size=256,
        num_channels=3,
        max_image_size=64,
    )
    full_cfg = HYV4FlashConfig(
        text_config=text_cfg.to_dict(), vision_config=vit_cfg.to_dict()
    )
    quant_cfg = Fp8Config.from_config(quant_dict)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = HYV4FlashForConditionalGeneration(
            config=full_cfg,
            mapping=Mapping(rank=0, world_size=1),
            quant_config=quant_cfg,
        )
    finally:
        torch.set_default_dtype(prev_dtype)

    l0 = model.language_model.model.layers[0]
    assert l0.self_attn.q_b_proj.weight.dtype == torch.bfloat16
    assert l0.self_attn.o_proj.weight.dtype == torch.bfloat16
    assert l0.self_attn.linear_gate.weight.dtype == torch.bfloat16
    assert l0.mlp.gate_up_proj.weight.dtype == torch.bfloat16

    l1 = model.language_model.model.layers[1]
    assert l1.self_attn.q_b_proj.weight.dtype == torch.float8_e4m3fn
    assert l1.self_attn.o_proj.weight.dtype == torch.float8_e4m3fn
    assert l1.self_attn.linear_gate.weight.dtype == torch.float8_e4m3fn
    assert l1.self_attn.q_a_proj.weight.dtype == torch.bfloat16
    assert l1.self_attn.kv_b_proj.weight.dtype == torch.bfloat16
    assert l1.mlp.shared_experts.gate_up_proj.weight.dtype == torch.bfloat16
    assert l1.mlp.routed_expert_down_proj.weight.dtype == torch.bfloat16
    assert l1.mlp.routed_expert_up_proj.weight.dtype == torch.bfloat16
    assert l1.mlp.experts.w13_weight.dtype == torch.float8_e4m3fn
    assert l1.mlp.experts.w2_weight.dtype == torch.float8_e4m3fn
    assert model.lm_head.weight.dtype == torch.float32

    weights: list[tuple[str, torch.Tensor]] = []
    for name, param in model.language_model.named_parameters():
        if ".mlp.experts." in name:
            continue
        if name.endswith(
            (".mlp.gate_up_proj.weight", ".shared_experts.gate_up_proj.weight")
        ):
            half = param.shape[0] // 2
            base = name[: -len("gate_up_proj.weight")]
            weights.append(
                (
                    f"{base}gate_proj.weight",
                    torch.zeros((half, param.shape[1]), dtype=param.dtype),
                )
            )
            weights.append(
                (
                    f"{base}up_proj.weight",
                    torch.zeros((half, param.shape[1]), dtype=param.dtype),
                )
            )
            continue
        if name.endswith("weight_scale_inv"):
            weights.append((name, torch.full(param.shape, 0.25, dtype=torch.bfloat16)))
        elif param.dtype == torch.float8_e4m3fn:
            weights.append((name, torch.zeros(param.shape, dtype=torch.float8_e4m3fn)))
        else:
            weights.append((name, torch.full(param.shape, 0.125, dtype=param.dtype)))

    for name, buf in model.language_model.named_buffers():
        if name.endswith("lookup_table"):
            weights.append((name, torch.randint(0, 16, buf.shape, dtype=buf.dtype)))

    weights.extend(
        [
            (
                "model.layers.1.mlp.experts.gate_up_proj",
                torch.zeros((4, 256, 128), dtype=torch.float8_e4m3fn),
            ),
            (
                "model.layers.1.mlp.experts.gate_up_proj_scale_inv",
                torch.full((4, 2, 1), 0.5, dtype=torch.bfloat16),
            ),
            (
                "model.layers.1.mlp.experts.down_proj",
                torch.zeros((4, 128, 128), dtype=torch.float8_e4m3fn),
            ),
            (
                "model.layers.1.mlp.experts.down_proj_scale_inv",
                torch.full((4, 1, 1), 0.75, dtype=torch.bfloat16),
            ),
        ]
    )
    for name, param in model.vit.named_parameters():
        weights.append(
            (f"vit.{name}", torch.full(param.shape, 0.25, dtype=param.dtype))
        )
    weights.append(
        (
            "model.dfly_layers.0.self_attn.q_a_proj.weight",
            torch.zeros((16, 16), dtype=torch.bfloat16),
        )
    )

    loaded = model.load_weights(weights)
    assert "model.dfly_layers.0.self_attn.q_a_proj.weight" not in loaded
    assert "vit.perceive.proj.0.weight" in loaded
    assert l1.self_attn.q_b_proj.weight_scale_inv.dtype == torch.float32
    assert torch.allclose(
        l1.self_attn.q_b_proj.weight_scale_inv,
        torch.full_like(l1.self_attn.q_b_proj.weight_scale_inv, 0.25),
    )
    assert l1.mlp.experts.w13_weight_scale_inv.dtype == torch.float32
    assert torch.allclose(
        l1.mlp.experts.w13_weight_scale_inv,
        torch.full_like(l1.mlp.experts.w13_weight_scale_inv, 0.5),
    )


def test_dfly2_117_tensor_fp8_checkpoint_loading():
    modules_to_not_convert = [
        "language_model.model.lm_head",
        "language_model.model.model.embed_tokens",
        "vit",
    ]
    for i in range(5):
        modules_to_not_convert.extend(
            [
                f"model.dfly_layers.{i}.self_attn.kv_a_proj_with_mqa",
                f"model.dfly_layers.{i}.self_attn.kv_b_proj",
                f"model.dfly_layers.{i}.self_attn.q_a_proj",
            ]
        )
    modules_to_not_convert.extend(
        [
            "model.dfly_layers.confidence_head.proj",
            "model.dfly_layers.hidden_correction.down_proj",
            "model.dfly_layers.hidden_correction.gate_proj",
            "model.dfly_layers.hidden_correction.up_proj",
            "model.dfly_layers.mtp_fusion.eh_proj",
        ]
    )
    quant_dict = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": modules_to_not_convert,
    }
    text_cfg = HYV4FlashTextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        routed_expert_hidden_size=128,
        num_hidden_layers=42,
        num_attention_heads=4,
        num_key_value_heads=4,
        swa_num_attention_heads=8,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        enable_lm_head_fp32=True,
        quantization_config=quant_dict,
    )
    full_cfg = HYV4FlashConfig(
        text_config=text_cfg.to_dict(),
        dfly_num_hidden_layers=5,
        dfly_intermediate_size=256,
        dfly_target_kv_layers=[25, 29, 33, 37, 41],
        dfly_target_index_layer=[25, 29, 33, 37, 41],
        dfly_target_index_topk=2048,
        dfly_mtp_fusion="slot0",
        dfly_enable_hidden_correction=True,
        dfly_enable_confidence_head=True,
        dfly_mask_token_id=250,
    )
    dfly_cfg = HYV4FlashDFlyConfig.from_target(full_cfg)
    quant_cfg = Fp8Config.from_config(quant_dict)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = HYV4FlashForCausalLMDFly(
            config=dfly_cfg,
            mapping=Mapping(rank=0, world_size=1),
            quant_config=quant_cfg,
        )
    finally:
        torch.set_default_dtype(prev_dtype)

    for layer in model.model.layers:
        assert layer.self_attn.q_b_proj.weight.dtype == torch.float8_e4m3fn
        assert layer.self_attn.o_proj.weight.dtype == torch.float8_e4m3fn
        assert layer.self_attn.linear_gate.weight.dtype == torch.float8_e4m3fn
        assert layer.mlp.gate_up_proj.weight.dtype == torch.float8_e4m3fn
        assert layer.mlp.down_proj.weight.dtype == torch.float8_e4m3fn
        assert layer.self_attn.q_a_proj.weight.dtype == torch.bfloat16
        assert layer.self_attn.kv_a_proj_with_mqa.weight.dtype == torch.bfloat16
        assert layer.self_attn.kv_b_proj.weight.dtype == torch.bfloat16

    weights: list[tuple[str, torch.Tensor]] = [
        (
            "model.layers.0.input_layernorm.weight",
            torch.ones(256, dtype=torch.bfloat16),
        ),
        ("vit.perceive.norm.weight", torch.ones(128, dtype=torch.bfloat16)),
    ]
    for i in range(5):
        p = f"model.dfly_layers.{i}"
        weights.extend(
            [
                (f"{p}.input_layernorm.weight", torch.ones(256, dtype=torch.bfloat16)),
                (
                    f"{p}.post_attention_layernorm.weight",
                    torch.ones(256, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.q_a_proj.weight",
                    torch.zeros((128, 256), dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.q_a_layernorm.weight",
                    torch.ones(128, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.q_b_proj.weight",
                    torch.zeros((512, 128), dtype=torch.float8_e4m3fn),
                ),
                (
                    f"{p}.self_attn.q_b_proj.weight_scale_inv",
                    torch.full((4, 1), 0.25, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.kv_a_proj_with_mqa.weight",
                    torch.zeros((192, 256), dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.kv_a_layernorm.weight",
                    torch.ones(128, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.kv_b_proj.weight",
                    torch.zeros((512, 128), dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.learnable_sink_param",
                    torch.zeros(4, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.linear_gate.weight",
                    torch.zeros((256, 256), dtype=torch.float8_e4m3fn),
                ),
                (
                    f"{p}.self_attn.linear_gate.weight_scale_inv",
                    torch.full((2, 2), 0.5, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.o_proj.weight",
                    torch.zeros((256, 256), dtype=torch.float8_e4m3fn),
                ),
                (
                    f"{p}.self_attn.o_proj.weight_scale_inv",
                    torch.full((2, 2), 0.5, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.self_attn.post_proj_norm.weight",
                    torch.ones(256, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.mlp.gate_proj.weight",
                    torch.zeros((256, 256), dtype=torch.float8_e4m3fn),
                ),
                (
                    f"{p}.mlp.gate_proj.weight_scale_inv",
                    torch.full((2, 2), 0.75, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.mlp.up_proj.weight",
                    torch.zeros((256, 256), dtype=torch.float8_e4m3fn),
                ),
                (
                    f"{p}.mlp.up_proj.weight_scale_inv",
                    torch.full((2, 2), 0.125, dtype=torch.bfloat16),
                ),
                (
                    f"{p}.mlp.down_proj.weight",
                    torch.zeros((256, 256), dtype=torch.float8_e4m3fn),
                ),
                (
                    f"{p}.mlp.down_proj.weight_scale_inv",
                    torch.full((2, 2), 0.25, dtype=torch.bfloat16),
                ),
            ]
        )
    weights.extend(
        [
            (
                "model.dfly_layers.confidence_head.proj.bias",
                torch.zeros(1, dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.confidence_head.proj.weight",
                torch.zeros((1, 256), dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.final_norm.weight",
                torch.ones(256, dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.hidden_correction.down_proj.weight",
                torch.zeros((256, 256), dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.hidden_correction.embed_norm.weight",
                torch.ones(256, dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.hidden_correction.gate_proj.weight",
                torch.zeros((256, 512), dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.hidden_correction.hidden_norm.weight",
                torch.ones(256, dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.hidden_correction.up_proj.weight",
                torch.zeros((256, 512), dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.mtp_fusion.eh_proj.weight",
                torch.zeros((256, 512), dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.mtp_fusion.enorm.weight",
                torch.ones(256, dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.mtp_fusion.gate",
                torch.zeros(256, dtype=torch.bfloat16),
            ),
            (
                "model.dfly_layers.mtp_fusion.hnorm.weight",
                torch.ones(256, dtype=torch.bfloat16),
            ),
        ]
    )
    assert sum(1 for k, _ in weights if k.startswith("model.dfly_layers.")) == 117

    loaded = model.load_weights(weights)
    assert len(loaded) == 107
    l0 = model.model.layers[0]
    assert l0.self_attn.q_b_proj.weight_scale_inv.dtype == torch.float32
    assert torch.allclose(
        l0.self_attn.q_b_proj.weight_scale_inv,
        torch.full((4, 1), 0.25, dtype=torch.float32),
    )
    assert l0.mlp.gate_up_proj.weight_scale_inv.dtype == torch.float32
    assert torch.allclose(
        l0.mlp.gate_up_proj.weight_scale_inv[:2],
        torch.full((2, 2), 0.75, dtype=torch.float32),
    )
    assert torch.allclose(
        l0.mlp.gate_up_proj.weight_scale_inv[2:],
        torch.full((2, 2), 0.125, dtype=torch.float32),
    )
