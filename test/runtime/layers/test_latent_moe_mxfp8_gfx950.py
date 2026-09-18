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

"""One-rank runtime lifecycle for the N16 bank; collectives are out of scope."""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from torch import nn

if not current_platform().is_cdna4:
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)


@pytest.fixture(scope="module")
def prepared_latent_experts():
    from tokenspeed_kernel.ops.moe import latent_moe_decode_pipeline_available

    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.layers.moe.latent import Kimi3LatentProjection
    from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            torch.device("cuda"),
            mock.patch.object(moe_utils, "MOE_BACKEND", moe_utils.MoeBackend.AUTO),
            mock.patch.object(
                moe_utils, "ALL2ALL_BACKEND", moe_utils.All2AllBackend.NONE
            ),
        ):
            experts = MoELayer(
                top_k=16,
                num_experts=896,
                hidden_size=3584,
                intermediate_size=3072,
                quant_config=Mxfp4Config(
                    ignored_layers=[],
                    is_checkpoint_mxfp4_serialized=True,
                    is_w4a8_fp8=True,
                    use_dynamic_mxfp4_activations=False,
                    quant_method="mxfp4",
                ),
                layer_index=1,
                tp_rank=0,
                tp_size=1,
                ep_rank=0,
                ep_size=8,
                activation="situ",
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                w13_input_layout="concatenated",
                routing_mode="precomputed_topk",
                internal_activation_dtype_override="input",
            )
            up = Kimi3LatentProjection(3584, 7168, params_dtype=torch.bfloat16)
            norm = RMSNorm(3584, eps=1e-6)
            shared_weight = torch.zeros((7168, 768), dtype=torch.bfloat16)
            projections = (
                torch.empty((896, 7168), dtype=torch.bfloat16),
                torch.empty((3584, 7168), dtype=torch.bfloat16),
                torch.empty((1536, 7168), dtype=torch.bfloat16),
                shared_weight,
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    assert experts.plan["internal_activation_dtype"] == "input"
    assert (
        experts.plan["apply_kernel_name"]
        == "gluon_mxfp4_a8w4_situ_ep_precomputed_moe_apply"
    )
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    assert latent_moe_decode_pipeline_available(
        *projections,
        *(getattr(experts, n) for n in names),
        experts.plan,
        topk=16,
        linear_clamp=25.0,
    )
    with torch.no_grad():
        for name in ("w13_weight_scale", "w2_weight_scale"):
            getattr(experts, name).fill_(125)
        for name in ("w13_weight", "w2_weight"):
            weight = getattr(experts, name)
            row = torch.arange(weight.shape[1], device="cuda")
            weight[0, row, row * 17 % weight.shape[2]] = (2 + row % 4).to(torch.uint8)
        row = torch.arange(7168, device="cuda")
        up.weight.zero_()
        up.weight[row, row % 3584] = 0.5
        shared_weight[row, row % 768] = 0.25

    # The loader invokes this real method on the already-constructed module.
    experts.process_weights_after_loading(experts)
    bank = tuple(getattr(experts, name) for name in names)
    assert all(t.ndim == 6 for t in bank)
    experts.process_weights_after_loading(experts)
    assert all(getattr(experts, name) is t for name, t in zip(names, bank, strict=True))
    assert latent_moe_decode_pipeline_available(
        *projections,
        *bank,
        experts.plan,
        topk=16,
        linear_clamp=25.0,
    )
    return experts, up, norm, shared_weight


@pytest.mark.parametrize("tokens", [1, 4, 33])
def test_n16_preprocess_standard_finalize_and_joint_decode(
    prepared_latent_experts, tokens
):
    from tokenspeed_kernel.ops.moe import latent_moe_expert_shared

    from tokenspeed.runtime.layers.moe.latent import LatentMoELayer
    from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput

    experts, up, norm, shared_weight = prepared_latent_experts
    hidden = torch.randn((tokens, 7168), device="cuda", dtype=torch.bfloat16) * 0.1
    routed_input = hidden[:, :3584].contiguous()
    shared_input = hidden[:, :768].contiguous()
    shared = shared_input @ shared_weight.T
    prefix = torch.randn_like(shared)
    saved_prefix = prefix.clone()
    logits = torch.empty((tokens, 896), device="cuda", dtype=torch.float32)
    ids = torch.full((tokens, 16), -1, device="cuda", dtype=torch.int32)
    weights = torch.zeros((tokens, 16), device="cuda", dtype=torch.float32)
    ids[:, 0], weights[:, 0] = 0, 1.0
    topk = StandardTopKOutput(weights, ids, logits)

    class FixedRouting(nn.Module):
        def forward(self, hidden_states, router_logits):
            return topk

    layer = LatentMoELayer(
        router=nn.Identity(),
        topk=FixedRouting(),
        routed_down_proj=nn.Identity(),
        experts=experts,
        routed_up_proj=up,
        routed_norm=norm,
        shared_experts=nn.Identity(),
        latent_reduce=lambda value: value,
        shared_reduce=None,
        joint_reduce=False,
        shared_expert_stream=None,
        expert_parallel_group=None,
        return_separate_outputs=False,
        input_projections=lambda values, output: (logits, routed_input, shared),
    )
    with torch.no_grad():
        routed = experts(routed_input, topk, tokens, tokens)
        expected = layer.finalize_output(routed, prefix, shared)
        actual = layer(
            hidden,
            num_global_tokens=tokens,
            max_num_tokens_per_gpu=tokens,
            prefix_sum=prefix,
        )
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
        )
        if tokens <= 4:
            arena = torch.empty(
                tokens * (7168 + 3584), device="cuda", dtype=torch.bfloat16
            )
            shared_out = arena[: tokens * 7168].view(tokens, 7168)
            routed_out = arena[tokens * 7168 :].view(tokens, 3584)
            joint_routed, joint_shared = latent_moe_expert_shared(
                routed_input,
                experts.w13_weight,
                experts.w13_weight_scale,
                experts.w2_weight,
                experts.w2_weight_scale,
                weights,
                ids,
                shared_input,
                shared_weight,
                activation_clamp=4.0,
                linear_clamp=25.0,
                expert_start=0,
                w13_interleaved=False,
                routed_out=routed_out,
                shared_out=shared_out,
            )
            assert joint_routed is routed_out and joint_shared is shared_out
            torch.testing.assert_close(
                joint_routed.view(torch.int16), routed.view(torch.int16), atol=0, rtol=0
            )
            torch.testing.assert_close(
                joint_shared.view(torch.int16), shared.view(torch.int16), atol=0, rtol=0
            )
            finished = layer.finalize_output(joint_routed, prefix, joint_shared)
            torch.testing.assert_close(
                finished.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
            )
    torch.testing.assert_close(prefix, saved_prefix, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("layout", ["concatenated", "interleaved"])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_n16_checkpoint_updates_preserve_loader_and_storage(device, layout, tp_rank):
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (
        preprocess_n16_mxfp4_weights,
    )

    from tokenspeed.runtime.layers.moe.loader import _load_fused_expert_tensor
    from tokenspeed.runtime.layers.moe.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.weights.mxfp4 import create_mxfp4_weight_pair

    spec = MoELayerSpec(
        top_k=2,
        num_experts=16,
        num_local_experts=2,
        hidden_size=256,
        intermediate_size=512,
        activation="situ",
        tp_rank=tp_rank,
        tp_size=2,
        ep_rank=1,
        ep_size=8,
        prefix="",
        a2a_backend="none",
    )
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    linear, packed = nn.Module(), nn.Module()
    for module in (linear, packed):
        module.w13_input_layout = layout
        with torch.device(device):
            create_mxfp4_weight_pair(spec, module, with_bias=False, solution="gluon")
    preprocess_n16_mxfp4_weights(packed)
    parameters = {name: getattr(packed, name) for name in names}
    pointers = {name: value.data_ptr() for name, value in parameters.items()}

    updates = (
        ("w13_weight", "w1", (16, 512, 128)),
        ("w13_weight", "w3", (16, 512, 128)),
        ("w2_weight", "w2", (16, 256, 256)),
        ("w13_weight_scale", "w1", (16, 512, 8)),
        ("w13_weight_scale", "w3", (16, 512, 8)),
        ("w2_weight_scale", "w2", (16, 256, 16)),
        ("w13_weight", "w13", (16, 1024, 128)),
        ("w13_weight_scale", "w13", (16, 1024, 8)),
    )
    for iteration, (name, shard, shape) in enumerate(updates):
        loaded = torch.arange(shape[0] * shape[1] * shape[2], device=device)
        loaded = ((loaded * 17 + iteration * 11) % 251).to(torch.uint8).reshape(shape)
        if name.endswith("scale"):
            loaded = loaded.view(torch.float8_e8m0fnu)
        for module in (linear, packed):
            _load_fused_expert_tensor(
                getattr(module, name),
                loaded,
                shard_id=shard,
                num_experts=16,
                ep_rank=1,
                ep_size=8,
            )
        expected = nn.Module()
        expected.w13_input_layout = layout
        for key in names:
            expected.register_parameter(
                key, nn.Parameter(getattr(linear, key).clone(), requires_grad=False)
            )
        preprocess_n16_mxfp4_weights(expected)
        for key in names:
            actual = getattr(packed, key)
            assert actual is parameters[key] and actual.data_ptr() == pointers[key]
            assert callable(actual.weight_loader)
            torch.testing.assert_close(actual, getattr(expected, key), atol=0, rtol=0)
