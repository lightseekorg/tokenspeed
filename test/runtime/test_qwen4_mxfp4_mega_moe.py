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


"""Qwen4 MXFP4 checkpoint loading and persistent MegaMoE planning."""

import os

import pytest
import torch
from tokenspeed_kernel.ops.moe import deep_gemm
from torch import nn

from tokenspeed.runtime.configs.qwen4_exp_config import Qwen4ExpTextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.layers.moe import utils as moe_utils
from tokenspeed.runtime.layers.moe.utils import All2AllBackend, MoeBackend
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.models.qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
    _mxfp4_mega_moe_capacity,
)
from tokenspeed.runtime.models.qwen4_exp import load_qwen4_exp_weights
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.mark.parametrize(
    "prefill,graph,requests,verify,expected",
    [
        (8192, 64, 256, 8, 8192),
        (128, 64, 256, 8, 512),
        (128, 16, 256, 8, 512),
        (128, 16, 64, 1, 128),
        (0, 16, 64, 8, 3072),
    ],
)
def test_capacity_includes_prefill_and_verify(
    monkeypatch, prefill, graph, requests, verify, expected
):
    mapping = Mapping(
        rank=0, world_size=8, attn_tp_size=2, moe_tp_size=1, moe_ep_size=8
    )
    for key, value in {
        "chunked_prefill_size": prefill,
        "prefill_graph_max_tokens": 0,
        "max_num_seqs": requests,
        "max_cudagraph_capture_size": graph,
        "speculative_num_draft_tokens": verify,
        "max_prefill_tokens": 1024,
        "max_model_len": 2048,
    }.items():
        monkeypatch.setitem(global_server_args_dict, key, value)
    assert _mxfp4_mega_moe_capacity(mapping) == expected


@pytest.mark.parametrize("max_model_len", [None, 0])
def test_unchunked_mxfp4_megamoe_requires_max_model_len(monkeypatch, max_model_len):
    mapping = Mapping(
        rank=0, world_size=8, attn_tp_size=2, moe_tp_size=1, moe_ep_size=8
    )
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", -1)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", max_model_len)
    with pytest.raises(ValueError, match="--max-model-len"):
        _mxfp4_mega_moe_capacity(mapping)


@pytest.mark.parametrize(
    "backend", [MoeBackend.MEGA_MOE, MoeBackend.DEEP_GEMM_MEGA_MOE]
)
@pytest.mark.parametrize("attn_tp", [1, 2])
def test_qwen4_mxfp4_plan_and_checkpoint(monkeypatch, backend, attn_tp):
    if not deep_gemm._HAS_DEEP_GEMM:
        pytest.skip("Blackwell DeepGEMM registry required; test tensors stay on CPU")
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", backend)
    monkeypatch.setattr(moe_utils, "ALL2ALL_BACKEND", All2AllBackend.NONE)
    monkeypatch.setattr(
        process_group_manager, "get_device_process_group", lambda group: None
    )
    mapping = Mapping(
        rank=2, world_size=4, attn_tp_size=attn_tp, moe_tp_size=1, moe_ep_size=4
    )
    for key, value in {
        "mapping": mapping,
        "chunked_prefill_size": 128,
        "prefill_graph_max_tokens": 0,
        "max_num_seqs": 32,
        "max_cudagraph_capture_size": 16,
        "speculative_num_draft_tokens": 4,
    }.items():
        monkeypatch.setitem(global_server_args_dict, key, value)
    config = Qwen4ExpTextConfig(
        hidden_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    )
    quant = Mxfp4Config.from_config(
        {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.0.mlp.shared_expert.gate_proj",
                "model.layers.0.mlp.shared_expert.up_proj",
                "model.layers.0.mlp.shared_expert.down_proj",
            ],
        }
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        block = Qwen3_5MoeSparseMoeBlock(
            config, mapping, quant, layer_index=0, prefix="model.layers.0.mlp"
        )
    finally:
        torch.set_default_dtype(previous)
    experts = block.experts
    assert experts.plan["apply_kernel_name"] == "deep_gemm_mxfp4_mega_moe_apply"
    assert experts.plan["persistent_max_num_tokens_per_gpu"] == 128
    assert block.shared_expert.gate_up_proj.weight.dtype == torch.bfloat16
    assert block.shared_expert.gate_up_proj.weight.shape == (256, 128)
    assert block.shared_expert.down_proj.weight.shape == (128, 128)
    container = nn.Module()
    container.model = nn.Module()
    layer = nn.Module()
    layer.mlp = block
    container.model.layers = nn.ModuleList([layer])
    weights = []
    for expert_id in range(8):
        for proj, rows, cols in [
            ("gate_proj", 128, 128),
            ("up_proj", 128, 128),
            ("down_proj", 128, 128),
        ]:
            prefix = f"model.language_model.layers.0.mlp.experts.{expert_id}.{proj}"
            weights += [
                (
                    f"{prefix}.weight",
                    torch.full((rows, cols // 2), expert_id + 1, dtype=torch.uint8),
                ),
                (
                    f"{prefix}.weight_scale",
                    torch.full((rows, cols // 32), 120 + expert_id, dtype=torch.uint8),
                ),
            ]
    load_qwen4_exp_weights(container, config, mapping, weights, include_visual=False)
    # Rank 2 owns experts 4 and 5; loading other ranks must not overwrite them.
    for local_id, expert_id in enumerate([4, 5]):
        assert torch.all(experts.w13_weight[local_id] == expert_id + 1)
        assert torch.all(experts.w2_weight[local_id] == expert_id + 1)
        assert torch.all(experts.w13_weight_scale[local_id] == 120 + expert_id)
        assert torch.all(experts.w2_weight_scale[local_id] == 120 + expert_id)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="Run with python -m torch.distributed.run --nproc-per-node=2",
)
@torch.no_grad()
def test_distributed_qwen4_shared_expert_forward(monkeypatch):
    import torch.distributed as dist
    import torch.nn.functional as F

    from tokenspeed.runtime.layers.linear import LinearBase

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    try:
        group = dist.group.WORLD
        monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.MEGA_MOE)
        monkeypatch.setattr(moe_utils, "ALL2ALL_BACKEND", All2AllBackend.NONE)
        monkeypatch.setattr(
            process_group_manager, "get_device_process_group", lambda _: group
        )
        mapping = Mapping(
            rank=group.rank(),
            world_size=group.size(),
            attn_tp_size=1,
            moe_tp_size=1,
            moe_ep_size=group.size(),
        )
        for key, value in {
            "mapping": mapping,
            "chunked_prefill_size": 64,
            "prefill_graph_max_tokens": 0,
            "max_num_seqs": 8,
            "max_cudagraph_capture_size": 8,
            "speculative_num_draft_tokens": 1,
        }.items():
            monkeypatch.setitem(global_server_args_dict, key, value)
        config = Qwen4ExpTextConfig(
            hidden_size=512,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
        )
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                block = Qwen3_5MoeSparseMoeBlock(
                    config,
                    mapping,
                    Mxfp4Config.from_config({"quant_method": "mxfp4"}),
                    layer_index=0,
                    prefix="model.layers.0.mlp",
                )
        finally:
            torch.set_default_dtype(previous)
        torch.manual_seed(19)
        for name, parameter in block.named_parameters():
            if parameter.dtype == torch.uint8:
                (
                    parameter.random_(120, 123)
                    if "scale" in name
                    else parameter.random_(0, 256)
                )
            else:
                parameter.normal_(0, 0.02)
        for module in block.modules():
            if isinstance(module, LinearBase):
                module.quant_method.process_weights_after_loading(module)
        block.experts.process_weights_after_loading(block.experts)
        block.warmup()
        assert block.experts._moe_backend_state.num_shared_experts == 0
        for rows in (7, 0):
            n = rows if group.rank() == 0 else rows + 3
            x = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
            # Obtain routed output through the real selected backend, then
            # independently compute the BF16 shared expert and sigmoid gate.
            logits, _ = block.gate(x)
            routes = (
                block.topk(x, logits)
                if n
                else block.topk.empty_topk_output(
                    x.device,
                    hidden_states=x,
                    router_logits=logits,
                )
            )
            routed = block.experts(
                hidden_states=x,
                topk_output=routes,
                num_global_tokens=rows * group.size() + 3 * (group.size() - 1),
                max_num_tokens_per_gpu=rows + 3,
                do_finalize=True,
            ).clone()
            gate, up = F.linear(x, block.shared_expert.gate_up_proj.weight).chunk(2, -1)
            shared = F.linear(
                (F.silu(gate.float()) * up.float()).to(x.dtype),
                block.shared_expert.down_proj.weight,
            )
            shared_gate = torch.sigmoid(
                F.linear(x.float(), block.shared_expert_gate.weight.float())
            )
            expected = (routed.float() + shared.float() * shared_gate).to(x.dtype)
            actual = block(
                x, rows * group.size() + 3 * (group.size() - 1), rows + 3, None
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
    finally:
        dist.destroy_process_group()
