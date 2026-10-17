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

"""Explicit Beam Gluon admission using real plans and storage-free weights."""

from test.runtime.models.beam_fixture import tiny_config
from test.runtime.models.test_beam_amd_fp8 import amd_platform  # noqa: F401
from test.runtime.models.test_beam_moe_quant import ct

import pytest
import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.models import beam

KERNEL = "gluon_fp8_channel_bf16_precomputed_moe_apply"
pytestmark = pytest.mark.usefixtures("amd_platform")


def make_experts(tp_size, **overrides):
    options = dict(
        top_k=8,
        num_experts=256,
        hidden_size=5120,
        intermediate_size=2560,
        quant_config=ct(),
        layer_index=1,
        prefix="model.layers.1.mlp.experts",
        tp_rank=0,
        tp_size=tp_size,
        ep_rank=0,
        ep_size=1,
        routing_mode="precomputed_topk",
    )
    options.update(overrides)
    with torch.device("meta"):
        return MoELayer(**options)


@pytest.mark.parametrize("tp_size", [4, 8])
def test_default_policy_selects_registered_gluon_and_preserves_storage(server, tp_size):
    server("gluon")
    experts = make_experts(tp_size)
    assert experts.plan["solution"] == "gluon"
    assert experts.plan["apply_kernel_name"] == KERNEL
    assert experts.plan.get("precision_policy", "default") == "default"
    assert experts.plan.get("output_scale", 1.0) == 1.0
    assert not experts.supports_deferred_finalize
    assert not experts.support_routing
    width = 2560 // tp_size
    assert experts.w13_weight.shape == (256, 2 * width, 5120)
    assert experts.w2_weight.shape == (256, 5120, width)
    assert experts.w13_weight.dtype == torch.float8_e4m3fn
    assert experts.w13_weight_scale.dtype == torch.float32


@pytest.mark.parametrize("tp_size", [4, 8])
def test_beam_explicit_selector_carries_default_policy_and_original_router(
    server, tp_size
):
    server("gluon")
    config = tiny_config()
    config.hidden_size = 5120
    config.moe_intermediate_size = 2560
    config.num_experts = 256
    config.num_experts_per_tok = 8
    config.n_group = 8
    config.topk_group = 4
    cls = beam.beam_moe_class(quantized_experts=True)
    assert cls is beam.BeamVendorMoE
    with torch.device("meta"):
        layer = cls(
            config,
            Mapping(rank=0, world_size=tp_size, moe_tp_size=tp_size, moe_ep_size=1),
            1,
            "model.layers.1.mlp",
            ct(),
        )
    assert layer.experts.plan["apply_kernel_name"] == KERNEL
    assert layer.experts.plan.get("precision_policy", "default") == "default"
    assert layer.route_scale == config.routed_scaling_factor
    assert layer.gate.weight.dtype == torch.float32
    assert layer.topk.topk_config.routed_scaling_factor is None
    assert beam.beam_moe_class(quantized_experts=False) is beam.BeamVendorMoE


@pytest.mark.parametrize("backend", ["triton", "auto"])
def test_default_policy_keeps_the_portable_fp8_backend(server, backend):
    server(backend)
    experts = make_experts(8)
    assert experts.plan["solution"] == "triton"
    assert (
        experts.plan["apply_kernel_name"]
        == "triton_fp8_channel_fused_precomputed_moe_apply"
    )
    assert experts.plan.get("precision_policy", "default") == "default"


@pytest.mark.parametrize(
    "overrides",
    [
        {"hidden_size": 128},
        {"intermediate_size": 2816},
        {"with_bias": True},
        {"routing_mode": "kernel_routing"},
        {"internal_activation_dtype_override": "fp8"},
    ],
)
def test_default_policy_refuses_unsupported_gluon_contract(server, overrides):
    from tokenspeed_kernel.selection import NoKernelFoundError

    server("gluon")
    with pytest.raises(NoKernelFoundError):
        make_experts(8, **overrides)


def test_gluon_beam_refuses_unsupported_unquantized_geometry(server):
    from tokenspeed_kernel.selection import NoKernelFoundError

    server("gluon")
    with pytest.raises(NoKernelFoundError):
        beam.BeamVendorMoE(
            tiny_config(), Mapping(rank=0, world_size=1), 1, "model.layers.1.mlp", None
        )


def test_default_gluon_policy_refuses_non_bf16_input(server):
    from tokenspeed_kernel.selection import NoKernelFoundError

    server("gluon")
    torch.set_default_dtype(torch.float32)
    with pytest.raises(NoKernelFoundError):
        make_experts(8)
