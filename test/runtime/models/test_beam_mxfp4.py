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

"""Beam's packed Quark MXFP4 admission, loading and precision policy."""

from test.runtime.models.beam_fixture import tiny_config, tiny_weights
from test.runtime.models.test_beam_amd_fp8 import amd_platform  # noqa: F401

import pytest
import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.models.beam import BeamForCausalLM
from tokenspeed.runtime.models.beam_quant import beam_quant_policy


def mxfp4_config():
    return Mxfp4Config(
        ignored_layers=[
            "re:.*self_attn.*",
            "re:.*mlp\\.gate",
            "re:model\\.layers\\.0\\..*",
            "lm_head",
            "model.embed_tokens",
        ],
        is_checkpoint_mxfp4_serialized=True,
        is_w4a8_fp8=False,
        use_dynamic_mxfp4_activations=True,
        quant_method="quark",
        fp8_layer_patterns=[],
    )


@pytest.mark.parametrize("world,rank", [(1, 0), (4, 0), (4, 3)])
def test_beam_mxfp4_keeps_packed_experts_and_bf16_dense_weights(
    server, amd_platform, world, rank
):
    server("triton")
    config = tiny_config()
    config.moe_intermediate_size = 512
    config.num_key_value_heads = 8
    checkpoint = tiny_weights(config)
    generator = torch.Generator().manual_seed(123)
    for name, weight in tuple(checkpoint.items()):
        if ".mlp.experts." not in name:
            continue
        rows, width = weight.shape
        checkpoint[name] = torch.randint(
            0, 256, (rows, width // 2), dtype=torch.uint8, generator=generator
        )
        checkpoint[f"{name}_scale"] = torch.full(
            (rows, width // 32), 120, dtype=torch.uint8
        )
    model = BeamForCausalLM(
        config, Mapping(rank=rank, world_size=world), mxfp4_config()
    )
    assert model.load_weights(checkpoint.items()) == checkpoint.keys()
    moe = model.model.layers[1].mlp
    experts = moe.experts
    assert experts.plan["precision_policy"] == "mxfp4_route_accum_fp32"
    assert experts.plan["apply_kernel_name"] == "triton_mxfp4_precomputed_moe_apply"
    assert experts.plan["output_scale"] == 2.5
    assert moe.route_scale is None and not experts.supports_deferred_finalize
    width = 512 // world
    channels = slice(rank * width, (rank + 1) * width)
    packed_channels = slice(rank * width // 2, (rank + 1) * width // 2)
    scale_channels = slice(rank * width // 32, (rank + 1) * width // 32)
    for expert in range(config.num_experts):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        for half, projection in enumerate(("gate_proj", "up_proj")):
            resident_rows = slice(half * width, (half + 1) * width)
            name = f"{prefix}.{projection}.weight"
            assert torch.equal(
                experts.w13_weight[expert, resident_rows], checkpoint[name][channels]
            )
            assert torch.equal(
                experts.w13_weight_scale[expert, resident_rows],
                checkpoint[f"{name}_scale"][channels],
            )
        name = f"{prefix}.down_proj.weight"
        assert torch.equal(
            experts.w2_weight[expert], checkpoint[name][:, packed_channels]
        )
        assert torch.equal(
            experts.w2_weight_scale[expert],
            checkpoint[f"{name}_scale"][:, scale_channels],
        )
    assert model.model.layers[0].mlp.gate_up_proj.weight.dtype == torch.bfloat16
    assert moe.gate.weight.dtype == torch.float32


@pytest.mark.parametrize(
    "fault", ["scale_shape", "scale_nan", "weight_dtype", "attention", "router"]
)
def test_beam_mxfp4_refuses_invalid_checkpoint_tensors(fault):
    policy = beam_quant_policy(mxfp4_config())
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    weight = torch.zeros((64, 64), dtype=torch.uint8)
    scale = torch.full((64, 4), 120, dtype=torch.uint8)
    if fault == "scale_shape":
        scale = scale[:, :2]
    elif fault == "scale_nan":
        scale[0, 0] = 255
    elif fault == "weight_dtype":
        weight = weight.float()
    elif fault == "attention":
        name = "model.layers.1.self_attn.q_proj.weight"
    elif fault == "router":
        name = "model.layers.1.mlp.gate.weight"
    with pytest.raises((ValueError, TypeError)):
        list(policy.adapt([(name, weight), (f"{name}_scale", scale)]))


def test_beam_mxfp4_refuses_weight_only_and_mixed_fp8_exports():
    config = mxfp4_config()
    config.use_dynamic_mxfp4_activations = False
    with pytest.raises(ValueError, match="serialized dynamic MXFP4"):
        beam_quant_policy(config)
    config.use_dynamic_mxfp4_activations = True
    config.fp8_layer_patterns = ["re:.*self_attn.*"]
    with pytest.raises(ValueError, match="serialized dynamic MXFP4"):
        beam_quant_policy(config)


def test_beam_mxfp4_refuses_unpacked_selected_experts():
    policy = beam_quant_policy(mxfp4_config())
    weight = torch.zeros((64, 128), dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="must be packed uint8"):
        list(policy.adapt([("model.layers.1.mlp.experts.0.gate_proj.weight", weight)]))


@pytest.mark.parametrize(
    "backend,ep", [("triton", 4), ("flashinfer_trtllm", 1), ("gluon", 1)]
)
def test_beam_mxfp4_refuses_unqualified_backends_and_expert_parallelism(
    server, amd_platform, backend, ep
):
    server(backend)
    config = tiny_config()
    config.moe_intermediate_size = 512
    config.num_key_value_heads = 8
    mapping = Mapping(rank=0, world_size=4, moe_ep_size=ep)
    with pytest.raises(
        ValueError, match="mxfp4_route_accum_fp32 requires|MXFP4 experts require"
    ):
        BeamForCausalLM(config, mapping, mxfp4_config())
