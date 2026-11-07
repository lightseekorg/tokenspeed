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

"""A tiny Beam checkpoint (random weights) for CPU and single-GPU tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tokenspeed.runtime.configs.beam_config import BeamConfig


def tiny_config() -> BeamConfig:
    return BeamConfig.from_dict(
        {
            "model_type": "beam",
            "architectures": ["BeamForCausalLM"],
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "hidden_act": "silu",
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": None,
            },
            "attention_bias": False,
            "attention_dropout": 0.0,
            "sliding_window": 64,
            "layer_types": ["sliding_attention", "full_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "moe_intermediate_size": 64,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "n_group": 2,
            "topk_group": 1,
            "routed_scaling_factor": 2.5,
            "final_logit_softcapping": 30.0,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 0,
            "torch_dtype": "bfloat16",
        }
    )


def tiny_weights(config: BeamConfig) -> dict[str, torch.Tensor]:
    """Deterministic checkpoint tensors under Beam's names, in their storage
    dtypes (the router FP32, everything else BF16)."""
    hidden = config.hidden_size
    query_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    shapes = {"model.embed_tokens.weight": ((config.vocab_size, hidden), None)}
    for layer_id in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_id}"
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            shapes[f"{prefix}.{norm}.weight"] = ((hidden,), None)
        for name, shape in (
            ("q_proj.weight", (query_width, hidden)),
            ("k_proj.weight", (kv_width, hidden)),
            ("v_proj.weight", (kv_width, hidden)),
            ("gate_proj.weight", (query_width, hidden)),
            ("o_proj.weight", (hidden, query_width)),
            ("gate_bias", (config.num_attention_heads,)),
            ("q_norm.weight", (config.head_dim,)),
            ("k_norm.weight", (config.head_dim,)),
        ):
            shapes[f"{prefix}.self_attn.{name}"] = (shape, None)
        mlp = f"{prefix}.mlp"
        if config.mlp_layer_types[layer_id] == "dense":
            branches = [(mlp, config.intermediate_size)]
        else:
            shapes[f"{mlp}.gate.weight"] = ((config.num_experts, hidden), torch.float32)
            shapes[f"{mlp}.gate.e_score_correction_bias"] = (
                (config.num_experts,),
                torch.float32,
            )
            branches = [
                (f"{mlp}.experts.{expert}", config.moe_intermediate_size)
                for expert in range(config.num_experts)
            ]
        for branch, width in branches:
            for name, shape in (
                ("gate_proj", (width, hidden)),
                ("up_proj", (width, hidden)),
                ("down_proj", (hidden, width)),
            ):
                shapes[f"{branch}.{name}.weight"] = (shape, None)
    shapes["model.norm.weight"] = ((hidden,), None)
    shapes["lm_head.weight"] = ((config.vocab_size, hidden), None)
    generator = torch.Generator(device="cpu").manual_seed(1729)
    weights = {}
    for name, (shape, dtype) in sorted(shapes.items()):
        value = torch.randn(shape, generator=generator) * 0.05
        if "norm.weight" in name:
            value += 1.0
        weights[name] = value.to(dtype or torch.bfloat16)
    return weights


def write_tiny_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    config = tiny_config()
    config.save_pretrained(directory)
    save_file(tiny_weights(config), str(directory / "model.safetensors"))
    vocabulary = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocabulary.update({f"t{index}": index for index in range(4, config.vocab_size)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast.save_pretrained(directory)
    (directory / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0}) + "\n"
    )
    return directory
