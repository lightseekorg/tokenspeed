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
    ExpertWeightShard,
    PreparedExpertWeights,
    create_peer_weight_store,
)

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.loader import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.peer_weights import MoEPeerWeights
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture(scope="module")
def device():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Launch with torchrun on at least two CUDA devices")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    yield device
    dist.destroy_process_group()


def _layer(quant, experts, index, manager):
    mxfp4 = quant == "mxfp4"
    # At 1 MiB per expert, remote experts stream through the staging slots.
    size = 1024 if mxfp4 else 256
    return MoELayer(
        top_k=4,
        num_experts=experts,
        hidden_size=size,
        intermediate_size=size,
        quant_config=(
            Mxfp4Config(ignored_layers=[], is_checkpoint_mxfp4_serialized=True)
            if mxfp4
            else Nvfp4Config()
        ),
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
        # The TRT-LLM MXFP4 SiTU kernel runs MXFP8 activations.
        internal_activation_dtype_override="fp8" if mxfp4 else None,
        peer_weights=manager,
    )


def _fill(layer, *, first, count, index):
    """Give every expert distinct weights and scales, by its global id."""
    with torch.no_grad():
        for name, parameter in layer.named_parameters(recurse=False):
            ids = torch.arange(first, first + count, device=parameter.device)
            if name.endswith("input_scale"):
                if name == "w13_input_scale":
                    parameter.zero_()
                else:
                    parameter.copy_(0.5 + ids * 0.001)
            elif name.endswith("weight_scale_2"):
                values = 0.01 + ids * 0.0001
                parameter.copy_(
                    values.reshape(count, *([1] * (parameter.ndim - 1))).expand_as(
                        parameter
                    )
                )
            elif name.endswith("weight_scale"):
                if parameter.dtype == torch.uint8:
                    # E8M0 exponents: block scales from 2**-9 to 2**-2.
                    parameter.copy_(
                        (118 + ids % 8)
                        .to(torch.uint8)
                        .reshape(count, *([1] * (parameter.ndim - 1)))
                        .expand_as(parameter)
                    )
                else:
                    parameter.fill_(0.125)
            elif name.endswith("weight"):
                shape = parameter.shape
                columns = torch.arange(parameter[0].numel(), device=parameter.device)
                parameter.copy_(
                    ((ids[:, None] * 13 + columns[None, :] + index * 7) % 255)
                    .reshape(shape)
                    .to(parameter.dtype)
                )


def _load_input_scales(layer, *, experts, rank, size):
    """Load every expert's input scale; the layer keeps the layer-wide maximum."""
    with torch.no_grad():
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


def _run_layers(layers, inputs, routing, manager):
    manager.prefetch_first()
    for layer in layers:
        inputs = layer(inputs, routing, inputs.shape[0], inputs.shape[0])
    return inputs


@pytest.mark.parametrize("num_layers", [3, 4])
def test_peer_manager_breakable_graph_replay(device, num_layers):
    partition = ExpertPartition(
        2 * dist.get_world_size(), dist.get_rank(), dist.get_world_size()
    )
    store = create_peer_weight_store(
        partition=partition,
        group=dist.group.WORLD,
        device=device,
        model_identity="peer-manager-graph-test",
        pool_page_bytes=16 * 1024**2,
        copy_implementation="simple",
        handle_type="posix_fd",
        solution="cuda",
    )
    manager = MoEPeerWeights(partition=partition, store=store)
    layers = []
    references = []
    # A whole VMM page per expert forces replay to refresh remote staging,
    # rather than reading only the immutable edge bytes populated at setup.
    width = store.granularity // 4
    try:
        for index in range(num_layers):
            full = (
                torch.arange(partition.num_experts, device=device, dtype=torch.float32)
                + index
                + 1
            )[:, None].expand(-1, width).contiguous() / 16
            layer = torch.nn.Linear(width, partition.count, bias=False, device=device)
            layer.weight.data.copy_(
                full[partition.first : partition.first + partition.count]
            )
            manager.publish(
                index,
                layer,
                PreparedExpertWeights(
                    "linear-test", (ExpertWeightShard("weight", layer.weight, False),)
                ),
            )
            layers.append(layer)
            references.append(full)
        manager.finalize()
        assert store.staging_bytes > 0
        inputs = torch.randn(2, width, device=device)

        def forward(value):
            manager.prefetch_first()
            outputs = []
            for index, layer in enumerate(layers):
                with manager.read(index):
                    outputs.append(layer(value))
            return torch.stack(outputs).sum(dim=0)

        with torch.no_grad():
            forward(inputs)
            capture = BreakableCapture()
            with capture:
                output = forward(inputs)
            for _ in range(8):
                inputs.normal_()
                expected = torch.stack(
                    [
                        torch.nn.functional.linear(inputs, weight)
                        for weight in references
                    ]
                ).sum(dim=0)
                capture.replay()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        manager.close()


@pytest.mark.parametrize("quant", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("num_layers", [3, 4])
def test_owned_loading_and_full_expert_compute(device, quant, num_layers):
    world, rank = dist.get_world_size(), dist.get_rank()
    partition = ExpertPartition(world * 2, rank, world)
    store = create_peer_weight_store(
        partition=partition,
        group=dist.group.WORLD,
        device=device,
        model_identity=f"runtime-moe-test-{quant}",
        pool_page_bytes=16 * 1024**2,
        copy_implementation="simple",
        handle_type="posix_fd",
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
            for index in range(num_layers):
                full = _layer(quant, partition.num_experts, index, None)
                own = _layer(quant, partition.num_experts, index, manager)
                assert own.w13_weight.shape[0] == partition.count
                assert own.num_local_experts == partition.num_experts
                _fill(full, first=0, count=partition.num_experts, index=index)
                _fill(own, first=partition.first, count=partition.count, index=index)
                if quant == "nvfp4":
                    _load_input_scales(
                        full, experts=partition.num_experts, rank=0, size=1
                    )
                    _load_input_scales(
                        own, experts=partition.num_experts, rank=rank, size=world
                    )
                full.process_weights_after_loading(full)
                own.process_weights_after_loading(own)
                resident.append(full)
                shared.append(own)
        manager.finalize()
        hidden_size = resident[0].hidden_size
        static_inputs = torch.empty(
            128, hidden_size, device=device, dtype=torch.bfloat16
        )
        static_weights = torch.empty(128, 4, device=device, dtype=torch.bfloat16)
        static_ids = torch.empty(128, 4, device=device, dtype=torch.int32)

        def routing():
            return StandardTopKOutput(static_weights, static_ids, None)

        def expected_output(inputs, route):
            for layer in resident:
                inputs = layer(inputs, route, inputs.shape[0], inputs.shape[0])
            return inputs

        # Warm up the real peer-reader path before capturing its eager breaks.
        static_inputs.normal_(std=0.2)
        static_weights.fill_(0.25)
        static_ids.copy_(
            (
                torch.arange(128, device=device)[:, None]
                + torch.arange(4, device=device)[None, :]
            )
            % partition.num_experts
        )
        warmup_route = routing()
        _run_layers(shared, static_inputs, warmup_route, manager)

        capture = BreakableCapture()
        with torch.no_grad():
            with capture:
                graph_output = _run_layers(shared, static_inputs, warmup_route, manager)

            # A direct replay is a smoke pass before serving changes static inputs.
            capture.replay()
            torch.testing.assert_close(
                graph_output,
                expected_output(static_inputs.clone(), warmup_route),
                rtol=0,
                atol=0,
            )

            # Exercise same-slot layer reuse and cross-slot prefetch over repeated
            # serving replays with live values copied into stable graph inputs.
            for iteration in range(8):
                inputs = torch.randn(
                    128, hidden_size, device=device, dtype=torch.bfloat16
                ) * (0.1 + iteration * 0.01)
                weights = torch.rand(128, 4, device=device, dtype=torch.bfloat16)
                ids = (
                    torch.arange(128, device=device)[:, None]
                    + torch.arange(4, device=device)[None, :]
                    + iteration
                ) % partition.num_experts
                static_inputs.copy_(inputs)
                static_weights.copy_(weights)
                static_ids.copy_(ids)
                expected = expected_output(inputs.clone(), routing())
                capture.replay()
                torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
    finally:
        manager.close()
        global_server_args_dict.clear()
        global_server_args_dict.update(saved)
        torch.set_default_dtype(dtype)
