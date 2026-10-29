# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.


import os
import sys
from dataclasses import replace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.mxfp4 import create_mxfp4_weight_pair
from tokenspeed.runtime.models import deepseek_v41 as v41

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")


class TestDeepseekV4MegaMoE:
    def test_standard_mxfp4_weights_load_expert_shards(self):
        experts = nn.Module()
        spec = MoELayerSpec(
            top_k=2,
            num_experts=4,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=128,
            activation="swiglu",
            tp_rank=0,
            tp_size=1,
            ep_rank=0,
            ep_size=2,
        )
        create_mxfp4_weight_pair(spec, experts)

        w1 = torch.full((128, 64), 1, dtype=torch.uint8)
        w3 = torch.full((128, 64), 3, dtype=torch.uint8)
        w2 = torch.full((128, 64), 2, dtype=torch.uint8)
        s1 = torch.full((128, 4), 11, dtype=torch.uint8)
        s3 = torch.full((128, 4), 13, dtype=torch.uint8)
        s2 = torch.full((128, 4), 12, dtype=torch.uint8)

        experts.w13_weight.weight_loader(
            experts.w13_weight, w1, "w1", local_expert_id=1
        )
        experts.w13_weight.weight_loader(
            experts.w13_weight, w3, "w3", local_expert_id=1
        )
        experts.w2_weight.weight_loader(experts.w2_weight, w2, "w2", local_expert_id=1)
        experts.w13_weight_scale.weight_loader(
            experts.w13_weight_scale, s1, "w1", local_expert_id=1
        )
        experts.w13_weight_scale.weight_loader(
            experts.w13_weight_scale, s3, "w3", local_expert_id=1
        )
        experts.w2_weight_scale.weight_loader(
            experts.w2_weight_scale, s2, "w2", local_expert_id=1
        )

        torch.testing.assert_close(experts.w13_weight[1, :128], w1)
        torch.testing.assert_close(experts.w13_weight[1, 128:], w3)
        torch.testing.assert_close(experts.w2_weight[1], w2)
        torch.testing.assert_close(experts.w13_weight_scale[1, :128], s1)
        torch.testing.assert_close(experts.w13_weight_scale[1, 128:], s3)
        torch.testing.assert_close(experts.w2_weight_scale[1], s2)


class _CudaWeight(nn.Parameter):
    """CPU storage satisfying the runtime's GPU-residency eligibility check."""

    @property
    def is_cuda(self):
        return True


def _mapping(*, dense_tp: int, moe_tp: int):
    from tokenspeed.runtime.distributed import Mapping

    world = 2 * moe_tp
    return Mapping(
        rank=0,
        world_size=world,
        attn_tp_size=1,
        attn_dp_size=world,
        dense_tp_size=dense_tp,
        dense_dp_size=world // dense_tp,
        moe_tp_size=moe_tp,
        moe_ep_size=2,
        moe_dp_size=1,
        nprocs_per_node=world,
        nnodes=1,
    )


@pytest.fixture
def shared_moe(monkeypatch):
    from tokenspeed_kernel.ops.moe import deep_gemm
    from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform

    from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41TextConfig
    from tokenspeed.runtime.layers.moe import expert, utils

    # These CPU model checks must select the CPU router on GPU CI hosts too.
    monkeypatch.setattr(
        Platform,
        "_instance",
        replace(current_platform(), vendor="cpu", arch_version=ArchVersion(0, 0)),
    )

    def cpu_plan(weight_dtype, **kwargs):
        # Selecting a native MegaMoE kernel requires GPU capability discovery.
        # Preserve the real constructor's numerical and geometry arguments.
        return dict(
            kwargs,
            apply_kernel_name="deep_gemm_mxfp4_mega_moe_apply",
            solution="mega_moe",
            topk_weights_dtype=torch.float32,
            support_routing=False,
        )

    monkeypatch.setattr(utils, "MOE_BACKEND", utils.MoeBackend.MEGA_MOE)
    monkeypatch.setattr(utils, "ALL2ALL_BACKEND", utils.All2AllBackend.NONE)
    monkeypatch.setattr(expert, "kernel_moe_plan", cpu_plan)
    monkeypatch.setattr(
        expert.pg_manager, "get_device_process_group", lambda group: None
    )
    mapping = _mapping(dense_tp=1, moe_tp=1)
    monkeypatch.setitem(v41.global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(v41.global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(v41.global_server_args_dict, "moe_mxfp4_fp8_activation", False)
    monkeypatch.setitem(v41.global_server_args_dict, "moe_combine_order", "rank")
    config = DeepseekV41TextConfig(
        hidden_size=256,
        moe_intermediate_size=128,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
        expert_dtype="fp4",
        swiglu_limit=10.0,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
    )
    quant = v41.v41_mxfp8_config(
        v41.Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
            scale_fmt="ue8m0",
        )
    )
    with v41.set_default_torch_dtype(torch.bfloat16):
        layer = v41.DeepseekV41MoE(config, mapping, quant, 0, "test.moe", None)
    with torch.no_grad():
        layer.gate.weight.zero_()
        for linear in (
            layer.shared_experts.gate_up_proj,
            layer.shared_experts.down_proj,
        ):
            linear.weight = _CudaWeight(
                torch.full(linear.weight.shape, 0.125, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            linear.weight_scale_inv.fill_(127)
            # The dense native prepacker is unavailable on CPU. Its owned
            # result is an eligibility prerequisite, not a shared input.
            linear.quant_method.load_as_bf16 = False
            linear.quant_method.prepared_weight_scales = linear.weight_scale_inv.clone()
    experts = layer.experts
    # Routed weight preparation requires CUDA. Supply its real state container;
    # shared preparation still runs the actual device-independent transforms.
    experts._moe_backend_state = deep_gemm._DeepGemmMegaMoEState(
        l1_weights=(experts.w13_weight, experts.w13_weight_scale),
        l2_weights=(experts.w2_weight, experts.w2_weight_scale),
        device=torch.device("cpu"),
    )
    experts._weights_processed = True
    return layer


def test_shared_post_load_prepares_dense_scales_before_fusion(shared_moe, monkeypatch):
    moe = shared_moe
    # Exercise the real post-load hook with just the affected MoE subtree.
    model = v41.DeepseekV41ForCausalLM.__new__(v41.DeepseekV41ForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Sequential(moe)
    shared = moe.shared_experts
    width = moe.config.moe_intermediate_size
    with torch.no_grad():
        shared.gate_up_proj.weight[:width].fill_(0.125)
        shared.gate_up_proj.weight[width:].fill_(0.25)
        shared.down_proj.weight.fill_(0.5)
    for linear in (shared.gate_up_proj, shared.down_proj):
        linear.quant_method.prepared_weight_scales = None
    monkeypatch.setattr(
        v41, "prepare_mxfp8_weight_scales", lambda weight, scales: scales.clone()
    )

    model.post_load_weights()

    prepared = moe.experts._moe_backend_state.shared
    assert prepared is not None
    assert prepared.intermediate_size == shared.down_proj.input_size == 128
    assert prepared.routed_intermediate_size == moe.experts.intermediate_size == 512
    # The first native block holds eight gate rows followed by eight up rows.
    assert torch.all(prepared.l1_weights[0][:8].float() == 0.125)
    assert torch.all(prepared.l1_weights[0][8:16].float() == 0.25)
    assert torch.all(prepared.l2_weights[0][:, :width].float() == 0.5)


def test_shared_preparation_requires_processed_routed_weights(shared_moe):
    shared_moe.experts._weights_processed = False
    with pytest.raises(RuntimeError, match="Routed weights must be processed"):
        shared_moe.prepare_shared_expert_fusion()
    assert shared_moe.experts._moe_backend_state.shared is None


@pytest.mark.parametrize(
    "condition",
    ["backend", "numerics", "dense_tp", "moe_tp", "gate_up_bias", "down_dtype"],
)
def test_ineligible_shared_expert_clears_prepared_output(
    shared_moe, monkeypatch, condition
):
    moe = shared_moe
    moe.prepare_shared_expert_fusion()
    assert moe.experts._moe_backend_state.shared is not None
    if condition == "backend":
        moe.use_mega_moe = False
    elif condition == "numerics":
        monkeypatch.setitem(v41.global_server_args_dict, "numerics", "rl-bitwise")
    elif condition == "dense_tp":
        moe.mapping = _mapping(dense_tp=2, moe_tp=1)
    elif condition == "moe_tp":
        moe.mapping = _mapping(dense_tp=1, moe_tp=2)
    elif condition == "gate_up_bias":
        linear = moe.shared_experts.gate_up_proj
        linear.bias = nn.Parameter(torch.zeros(linear.weight.shape[0]))
    else:
        linear = moe.shared_experts.down_proj
        linear.weight = _CudaWeight(
            linear.weight.to(torch.bfloat16), requires_grad=False
        )

    moe.prepare_shared_expert_fusion()

    assert moe.experts._moe_backend_state.shared is None


@pytest.mark.parametrize("tokens", [0, 3])
def test_shared_forward_adds_once_and_restores_fallback(
    shared_moe, monkeypatch, tokens
):
    from tokenspeed_kernel.selection import kernel_override

    from tokenspeed.runtime.layers.dense import fp8
    from tokenspeed.runtime.layers.moe import expert

    moe = shared_moe

    def cpu_fp8_mm(
        a, b, *, A_scales, B_scales, bias, out_dtype, quant, block_size, solution, out
    ):
        # Replace the GPU GEMM; the real shared MLP and activation quantizer run.
        a_scale = (A_scales.to(torch.int32) << 23).view(torch.float32)
        b_scale = (B_scales.to(torch.int32) << 23).view(torch.float32)
        a = a.float() * a_scale.repeat_interleave(32, dim=-1)
        b = b.float() * b_scale.repeat_interleave(32, dim=-1)
        result = torch.nn.functional.linear(a, b, bias).to(out_dtype)
        return result if out is None else out.copy_(result)

    monkeypatch.setattr(fp8, "mm", cpu_fp8_mm)
    hidden = torch.full((tokens, moe.config.hidden_size), 0.03125, dtype=torch.bfloat16)
    input_ids = torch.zeros(tokens, dtype=torch.int64)
    shared, routed = moe.shared_experts(hidden), hidden * 2
    expected = routed + shared
    launches = []

    def cpu_moe_apply(plan, x, weights, router_logits, **kwargs):
        launches.append(x.shape[0])
        return expected if weights._moe_backend_state.shared is not None else routed

    monkeypatch.setattr(expert, "kernel_moe_apply", cpu_moe_apply)
    shared_calls = []
    with moe.shared_experts.register_forward_hook(lambda *args: shared_calls.append(1)):
        moe.prepare_shared_expert_fusion()
        # An empty local rank still participates when another rank has tokens.
        global_tokens = max(tokens, 1)
        combined = moe.forward_mega_moe(
            hidden, input_ids, global_tokens, global_tokens, None, None
        )
        torch.testing.assert_close(combined, expected, rtol=0, atol=0)
        assert shared_calls == []

        with kernel_override("gemm", "mm", "test_override"):
            moe.prepare_shared_expert_fusion()
            assert moe.experts._moe_backend_state.shared is None
            standalone = moe.forward_mega_moe(
                hidden, input_ids, global_tokens, global_tokens, None, None
            )
        torch.testing.assert_close(standalone, expected, rtol=0, atol=0)
        assert shared_calls == ([1] if tokens else [])
    assert launches == [tokens, tokens]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
