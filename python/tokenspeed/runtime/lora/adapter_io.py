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

"""PEFT LoRA adapter loading and metadata helpers."""

from __future__ import annotations

import json
import os
import re

import torch

PEFT_ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
PEFT_MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
PEFT_EXPERT_MODULES = PEFT_MLP_MODULES
PEFT_HEAD_MODULE = "lm_head"
PEFT_MODULES = (*PEFT_ATTN_MODULES, *PEFT_MLP_MODULES)

# Sentinel layer_id used for model-global modules (e.g. lm_head) that have no
# per-layer index in AdapterWeights.
LORA_HEAD_LAYER_ID = -1

AdapterWeights = dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]


def resolve_adapter_weight_path(adapter_path: str) -> str:
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    return safetensors_path if os.path.exists(safetensors_path) else adapter_path


def load_adapter_weights(adapter_path: str) -> AdapterWeights:
    return parse_adapter_weights(
        load_safetensors(resolve_adapter_weight_path(adapter_path))
    )


def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def parse_adapter_weights(tensors: dict[str, torch.Tensor]) -> AdapterWeights:
    """Return ``{layer_id: {module_name: (lora_A, lora_B)}}``.

    Matches attention (``self_attn.{q,k,v,o}_proj``), MLP
    (``mlp.{gate,up,down}_proj``), and lm_head PEFT module names.
    lm_head weights are stored under ``LORA_HEAD_LAYER_ID`` (-1).
    """
    dense_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\."
        r"(?:self_attn|mlp)\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
        r"lora_(A|B)\.weight"
    )
    expert_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\."
        r"mlp\.experts\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)\."
        r"lora_(A|B)\.weight"
    )
    expert_3d_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\."
        r"mlp\.experts\."
        r"(w1|w2|w3)\."
        r"lora_(A|B)\.weight"
    )
    # PEFT uses ``lora_embedding_A/B`` (no ``.weight`` suffix) for modules
    # treated as embedding tables (lm_head, embed_tokens).
    head_pattern = re.compile(
        r"base_model\.model\.lm_head\." r"(?:lora_(A|B)\.weight|lora_embedding_(A|B))"
    )
    weights: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for key, tensor in tensors.items():
        m = dense_pattern.match(key)
        if m:
            layer_id, module, ab = int(m.group(1)), m.group(2), m.group(3)
        else:
            m = expert_pattern.match(key)
            if m:
                layer_id = int(m.group(1))
                module = f"experts.{int(m.group(2))}.{m.group(3)}"
                ab = m.group(4)
            else:
                m = expert_3d_pattern.match(key)
                if m:
                    layer_id = int(m.group(1))
                    module = f"experts.{m.group(2)}"
                    ab = m.group(3)
                else:
                    m = head_pattern.match(key)
                    if not m:
                        continue
                    layer_id = LORA_HEAD_LAYER_ID
                    module = PEFT_HEAD_MODULE
                    ab = m.group(1) or m.group(2)
        weights.setdefault(layer_id, {}).setdefault(module, {})[ab] = tensor

    result: AdapterWeights = {}
    for layer_id, modules in weights.items():
        result[layer_id] = {}
        for module, ab_dict in modules.items():
            result[layer_id][module] = (ab_dict["A"], ab_dict["B"])
    return result


def read_adapter_scaling(adapter_path: str | None, rank: int) -> float:
    if adapter_path is None:
        return 1.0
    config_file = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_file):
        return 1.0
    try:
        with open(config_file) as f:
            cfg = json.load(f)
        alpha = float(cfg.get("lora_alpha", rank))
        r = int(cfg.get("r", rank))
        return alpha / r if r > 0 else 1.0
    except Exception:
        return 1.0
