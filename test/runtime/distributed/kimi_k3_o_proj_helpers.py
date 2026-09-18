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

"""Shared mapping, weight loading, and linear construction for projection checks."""

import json
import os
from pathlib import Path

import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.o_proj import (
    make_output_projection,
    projection_mapping,
)
from tokenspeed.runtime.utils.env import envs

ENV_NAME = envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE.name


def dep_mapping(rank: int, world: int) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dp_size=world,
        attn_dcp_size=1,
        dense_tp_size=1,
        dense_dp_size=world,
        moe_tp_size=1,
        moe_ep_size=world,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=1,
        linear_attn_tp_size=1,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=None,
        nnodes=None,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def load_projection(model: str, layer: int):
    from safetensors import safe_open

    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

    root = Path(model)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = f"language_model.model.layers.{layer}.self_attn.o_proj"
    with safe_open(root / index[name + ".weight"], framework="pt", device="cpu") as f:
        weight = f.get_tensor(name + ".weight")
    with safe_open(
        root / index[name + ".weight_scale"], framework="pt", device="cpu"
    ) as f:
        scale = f.get_tensor(name + ".weight_scale").squeeze()
    assert (
        weight.dtype == torch.float8_e4m3fn
    ), "Harness expects checkpoint FP8 attention weights"
    quant = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[],
        weight_block_size=[128, 128],
        scale_fmt=None,
    )
    return weight, scale, quant


def make_linears(mapping: Mapping, weight, scale, quant):
    k, n = weight.shape[1], weight.shape[0]
    result = []
    for size in (1, 4):
        os.environ[ENV_NAME] = str(size)
        with torch.device("cuda"):
            linear, exchange = make_output_projection(
                parallel=projection_mapping(mapping.rank, mapping.world_size, size),
                input_size=k,
                output_size=n,
                quant_config=quant,
                prefix="self_attn.o_proj",
                default_parallel=mapping.attn,
                reduce_results=False,
            )
        linear.weight.weight_loader(linear.weight, weight)
        if scale is not None:
            linear.weight_scale_inv.weight_loader(linear.weight_scale_inv, scale)
        linear.quant_method.process_weights_after_loading(linear)
        result.append((linear, exchange))
    os.environ[ENV_NAME] = "4"
    return result
