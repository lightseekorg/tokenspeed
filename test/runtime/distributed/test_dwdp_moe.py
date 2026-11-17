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

"""Real MoELayer loading and forward with sharded storage; launch under torchrun."""
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.peer_weights import (
    ExpertPartition,
    create_peer_weight_store,
)

from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.loader import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.peer_weights import MoEPeerWeights
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.utils.env import global_server_args_dict


def _layer(experts, index, manager):
    return MoELayer(
        top_k=4,
        num_experts=experts,
        hidden_size=256,
        intermediate_size=256,
        quant_config=Nvfp4Config(),
        layer_index=index,
        prefix=f"layers.{index}.mlp",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
        activation="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        routing_config={},
        routing_mode="precomputed_topk",
        peer_weights=manager,
    )


def _load(layer, *, first, count, index, experts, rank, size):
    with torch.no_grad():
        for name, parameter in layer.named_parameters(recurse=False):
            if name.endswith("input_scale"):
                if name == "w13_input_scale":
                    parameter.zero_()
                else:
                    parameter.copy_(
                        0.5
                        + torch.arange(first, first + count, device=parameter.device)
                        * 0.001
                    )
            elif name.endswith("weight_scale_2"):
                values = (
                    0.01
                    + torch.arange(first, first + count, device=parameter.device)
                    * 0.0001
                )
                parameter.copy_(
                    values.reshape(count, *([1] * (parameter.ndim - 1))).expand_as(
                        parameter
                    )
                )
            elif name.endswith("weight_scale"):
                parameter.fill_(0.125)
            elif name.endswith("weight"):
                shape = parameter.shape
                columns = torch.arange(parameter[0].numel(), device=parameter.device)
                ids = torch.arange(first, first + count, device=parameter.device)
                parameter.copy_(
                    ((ids[:, None] * 13 + columns[None, :] + index * 7) % 255)
                    .reshape(shape)
                    .to(parameter.dtype)
                )
        loader = build_moe_checkpoint_loader(
            params_dict={
                f"experts.{name}": value for name, value in layer.named_parameters()
            },
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1", up_proj_name="w3", down_proj_name="w2"
            ),
            num_experts=experts,
            ep_rank=rank,
            ep_size=size,
        )
        for expert in range(experts):
            loader.load(
                f"experts.{expert}.w1.input_scale",
                torch.tensor([1 + expert * 0.001], device="cuda", dtype=torch.float32),
            )
        torch.testing.assert_close(
            layer.w13_input_scale,
            torch.tensor(
                [1 + (experts - 1) * 0.001], device="cuda", dtype=torch.float32
            ),
            rtol=1e-6,
            atol=0,
        )


def test_owned_loading_and_full_expert_compute():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Launch with torchrun on fabric-connected GPUs")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    world, rank = dist.get_world_size(), dist.get_rank()
    partition = ExpertPartition(world * 2, rank, world)
    store = create_peer_weight_store(
        partition=partition,
        group=dist.group.WORLD,
        device=device,
        model_identity="runtime-moe-test",
        pool_page_bytes=16 * 1024**2,
        copy_implementation="simple",
        solution="cuda",
    )
    manager = MoEPeerWeights(partition=partition, store=store)
    saved = global_server_args_dict.copy()
    dtype = torch.get_default_dtype()
    global_server_args_dict.update(
        moe_backend="flashinfer_trtllm", all2all_backend="none"
    )
    torch.set_default_dtype(torch.bfloat16)
    try:
        resident, shared = [], []
        with torch.device(device):
            for index in range(3):
                full = _layer(partition.num_experts, index, None)
                own = _layer(partition.num_experts, index, manager)
                assert own.w13_weight.shape[0] == partition.count
                assert own.num_local_experts == partition.num_experts
                _load(
                    full,
                    first=0,
                    count=partition.num_experts,
                    index=index,
                    experts=partition.num_experts,
                    rank=0,
                    size=1,
                )
                _load(
                    own,
                    first=partition.first,
                    count=partition.count,
                    index=index,
                    experts=partition.num_experts,
                    rank=rank,
                    size=world,
                )
                full.process_weights_after_loading(full)
                own.process_weights_after_loading(own)
                resident.append(full)
                shared.append(own)
        manager.finalize()
        inputs = torch.randn(128, 256, device=device, dtype=torch.bfloat16) * 0.2
        ids = (
            (
                torch.arange(128, device=device)[:, None]
                + torch.arange(4, device=device)[None, :]
            )
            % partition.num_experts
        ).int()
        routing = StandardTopKOutput(
            torch.full((128, 4), 0.25, device=device, dtype=torch.bfloat16), ids, None
        )
        expected = [layer(inputs, routing, 128, 128) for layer in resident]
        for iteration in range(24):
            manager.prefetch_first()
            for index, layer in enumerate(shared):
                actual = layer(inputs, routing, 128, 128)
                torch.testing.assert_close(actual, expected[index], rtol=0, atol=0)
    finally:
        manager.close()
        global_server_args_dict.clear()
        global_server_args_dict.update(saved)
        torch.set_default_dtype(dtype)
        dist.destroy_process_group()
