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

"""Beam AMD admission and checkpoint loading; no GPU kernels are launched."""

import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform
from tokenspeed_kernel.registry import KernelRegistry

from tokenspeed.runtime.distributed import comm_manager as comm_module
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.models import beam
from tokenspeed.runtime.models.beam_sandwich_fusion import (
    BeamSandwichNorm,
    beam_sandwich_fusion_refusal,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

# The CI runner executes registered test files as standalone scripts.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.models.beam_fixture import tiny_config, tiny_weights  # noqa: E402
from test.runtime.models.test_beam_moe_quant import (  # noqa: E402
    ct,
    fp8_checkpoint,
    quantize,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=40, suite="runtime-1gpu")


@pytest.mark.parametrize("mixed", [False, True])
def test_beam_scheduler_preserves_nonmixed_capacity_and_explicit_opt_in(mixed):
    from tokenspeed.runtime.engine.rank_setup import scheduler_config_from_args
    from tokenspeed.runtime.utils.server_args import ServerArgs

    args = ServerArgs(
        model="x",
        max_num_seqs=16,
        chunked_prefill_size=8192,
        enable_mixed_batch=mixed,
    )
    # Device construction requires GPUs; only its cache geometry is needed here.
    specs = SimpleNamespace(
        cache_geometry=SimpleNamespace(num_device_pages=64, prefix_granularity=16),
        cache_groups=[],
        num_host_pages=17,
        num_snapshot_pages=1,
        max_retracted_requests=0,
    )
    config = scheduler_config_from_args(
        args,
        specs,
        # No storage backend is configured: admission passes no storage limits.
        storage_config=None,
        max_scheduled_tokens=args.chunked_prefill_size,
        max_batch_size=args.max_num_seqs,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        enable_kv_cache_events=False,
        prefix_replay_tokens=0,
    )
    assert config.enable_mixed_prefill_decode is mixed
    assert config.max_batch_size == 16
    assert config.max_scheduled_tokens == 8192


@pytest.fixture
def amd_platform():
    original = current_platform()
    Platform.override(
        dataclasses.replace(original, vendor="amd", arch_version=ArchVersion(9, 5))
    )
    yield original
    Platform.override(original)


def _fp8_layer():
    cls = beam.beam_moe_class(quantized_experts=True)
    return cls(
        tiny_config(), Mapping(rank=0, world_size=1), 1, "model.layers.1.mlp", ct()
    )


@pytest.mark.parametrize("fmt", ["bf16", "fp8_channel", "mxfp4"])
@pytest.mark.parametrize("backend", ["auto", "triton", "gluon"])
@pytest.mark.parametrize("tp_size", [4, 8])
def test_amd_default_formats_select_registry_backend_and_scale_routes_once(
    server, amd_platform, fmt, backend, tp_size
):
    from test.runtime.models.test_beam_mxfp4 import mxfp4_config

    expected_backend = backend if backend != "auto" else "gluon"
    # These Gluon registrations are loaded only on AMD. Keep portable
    # formats/backends covered when the process was imported on another vendor.
    amd_registration = {
        "bf16": "gluon_bf16_precomputed_moe_apply",
        "mxfp4": "gluon_mxfp4_precomputed_moe_apply",
    }.get(fmt)
    if (
        not amd_platform.is_amd
        and expected_backend == "gluon"
        and amd_registration is not None
        and KernelRegistry.get().get_by_name(amd_registration) is None
    ):
        pytest.skip(f"AMD backend {amd_registration!r} is not registered")

    server(backend)
    config = tiny_config()
    config.hidden_size = 5120
    config.moe_intermediate_size = 2560
    config.num_experts = 256
    config.num_experts_per_tok = 8
    config.n_group = 8
    config.topk_group = 4
    quant = {"bf16": None, "fp8_channel": ct(), "mxfp4": mxfp4_config()}[fmt]
    cls = beam.beam_moe_class(quantized_experts=quant is not None)
    with torch.device("meta"):
        layer = cls(
            config, Mapping(rank=0, world_size=tp_size), 1, "model.layers.1.mlp", quant
        )
    experts = layer.experts
    assert experts.plan["solution"] == expected_backend
    assert experts.plan.get("precision_policy", "default") == "default"
    assert experts.plan.get("output_scale", 1.0) == 1.0
    assert experts.supports_precomputed_topk
    assert not experts.support_routing and not experts.supports_deferred_finalize
    assert layer.gate.weight.dtype == torch.float32
    layer.gate.e_score_correction_bias = torch.nn.Parameter(
        torch.zeros(256, dtype=torch.float32)
    )
    logits = torch.linspace(-2, 2, 512, dtype=torch.float32).view(2, 256)
    actual = layer.topk(torch.zeros(2, 5120), logits)
    expected, ids = beam.beam_route_weights(
        logits, layer.gate.e_score_correction_bias, 8, True, 8, 4, None
    )
    assert torch.equal(actual.topk_ids, ids)
    assert actual.topk_weights.dtype == torch.float32
    assert torch.equal(actual.topk_weights, expected * config.routed_scaling_factor)
    assert layer.topk.topk_config.routed_scaling_factor is None


@pytest.mark.parametrize("backend", ["auto", "triton", "gluon"])
def test_amd_default_still_refuses_nvfp4(server, amd_platform, backend):
    from test.runtime.models.test_beam_moe_quant import NVFP4

    server(backend)
    with pytest.raises(ValueError, match="NVFP4 experts require NVIDIA"):
        beam.BeamVendorMoE(
            tiny_config(), Mapping(rank=0, world_size=1), 1, "model.layers.1.mlp", NVFP4
        )


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_nvidia_default_keeps_bf16_policy_and_late_scale(server, amd_platform, backend):
    server(backend)
    Platform.override(
        dataclasses.replace(
            current_platform(), vendor="nvidia", arch_version=ArchVersion(10, 3)
        )
    )
    cls = beam.beam_moe_class()
    assert cls is beam.BeamMoE
    layer = cls(tiny_config(), Mapping(rank=0, world_size=1), 1, "model.layers.1.mlp")
    assert layer.experts.plan["precision_policy"] == "bf16_route_accum_fp32"
    assert layer.experts.plan["output_scale"] == 2.5
    assert layer.route_scale is None
    assert (
        layer.experts.plan["apply_kernel_name"] == "triton_bf16_precomputed_moe_apply"
    )
    with pytest.raises(ValueError, match="flashinfer_trtllm"):
        beam.beam_moe_class(quantized_experts=True)


def test_amd_fp8_and_bf16_admission_share_the_default_path(server, amd_platform):
    server("triton")
    assert beam.beam_moe_class(quantized_experts=True) is beam.BeamVendorMoE
    assert beam.beam_moe_class(quantized_experts=False) is beam.BeamVendorMoE


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_amd_fp8_layer_keeps_checkpoint_codes_and_channel_scales(
    server, amd_platform, backend
):
    server(backend)
    layer = _fp8_layer()
    experts = layer.experts
    assert experts.plan.get("precision_policy", "default") == "default"
    assert (
        experts.plan["apply_kernel_name"]
        == "triton_fp8_channel_fused_precomputed_moe_apply"
    )
    assert experts.plan["solution"] == "triton"
    assert not experts.supports_deferred_finalize and not experts.support_routing
    assert experts.w13_weight.dtype == torch.float8_e4m3fn
    assert experts.w2_weight.dtype == torch.float8_e4m3fn
    assert experts.w13_weight_scale.dtype == torch.float32
    assert experts.w13_weight_scale.shape == (8, 128)
    assert experts.w2_weight_scale.shape == (8, 128)
    assert layer.route_scale == 2.5
    assert layer.topk.topk_config.routed_scaling_factor is None


def test_amd_fp8_loader_retains_resident_fp8_weights(server, amd_platform):
    server("triton")
    model = beam.BeamForCausalLM(tiny_config(), Mapping(rank=0, world_size=1), ct())
    checkpoint, _ = fp8_checkpoint()
    assert model.load_weights(checkpoint.items()) == checkpoint.keys()
    experts = model.model.layers[1].mlp.experts
    for expert in range(8):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        for half, projection in enumerate(("gate_proj", "up_proj")):
            rows = slice(half * 64, (half + 1) * 64)
            assert torch.equal(
                experts.w13_weight[expert, rows].view(torch.uint8),
                checkpoint[f"{prefix}.{projection}.weight"].view(torch.uint8),
            )
            assert torch.equal(
                experts.w13_weight_scale[expert, rows],
                checkpoint[f"{prefix}.{projection}.weight_scale"].float().flatten(),
            )
        assert torch.equal(
            experts.w2_weight[expert].view(torch.uint8),
            checkpoint[f"{prefix}.down_proj.weight"].view(torch.uint8),
        )


@pytest.mark.parametrize("world,rank", [(4, 0), (4, 3), (8, 0), (8, 7)])
@pytest.mark.parametrize("expert_parallel", [False, True])
def test_amd_fp8_checkpoint_partitions_codes_and_scales(
    server, amd_platform, world, rank, expert_parallel
):
    server("triton")
    config = tiny_config()
    config.moe_intermediate_size = 512
    config.num_key_value_heads = 8
    checkpoint = tiny_weights(config)
    for name, weight in tuple(checkpoint.items()):
        if name.endswith("proj.weight") and weight.ndim == 2:
            codes, scales = quantize(weight)
            checkpoint[name] = codes
            checkpoint[f"{name.removesuffix('.weight')}.weight_scale"] = scales
    mapping = Mapping(
        rank=rank, world_size=world, moe_ep_size=world if expert_parallel else 1
    )
    model = beam.BeamForCausalLM(config, mapping, ct())
    assert model.load_weights(checkpoint.items()) == checkpoint.keys()
    experts = model.model.layers[1].mlp.experts
    local_experts = 8 // world if expert_parallel else 8
    width = 512 if expert_parallel else 512 // world
    start_expert = rank * local_experts if expert_parallel else 0
    start_channel = 0 if expert_parallel else rank * width
    channels = slice(start_channel, start_channel + width)
    assert experts.num_local_experts == local_experts
    for local in range(local_experts):
        prefix = f"model.layers.1.mlp.experts.{start_expert + local}"
        for half, projection in enumerate(("gate_proj", "up_proj")):
            rows = slice(half * width, (half + 1) * width)
            assert torch.equal(
                experts.w13_weight[local, rows].view(torch.uint8),
                checkpoint[f"{prefix}.{projection}.weight"][channels].view(torch.uint8),
            )
            assert torch.equal(
                experts.w13_weight_scale[local, rows],
                checkpoint[f"{prefix}.{projection}.weight_scale"][channels]
                .float()
                .flatten(),
            )
        assert torch.equal(
            experts.w2_weight[local].view(torch.uint8),
            checkpoint[f"{prefix}.down_proj.weight"][:, channels]
            .contiguous()
            .view(torch.uint8),
        )
        assert torch.equal(
            experts.w2_weight_scale[local],
            checkpoint[f"{prefix}.down_proj.weight_scale"].float().flatten(),
        )


@pytest.mark.parametrize(
    "change", [{"output_scale": 2.0}, {"precision_policy": "fp8ch_vendor_fp32_route"}]
)
def test_portable_fp8_layer_refuses_an_incompatible_precision_contract(
    server, amd_platform, change
):
    server("triton")
    with pytest.raises(ValueError, match="requires"):
        expert_module.MoELayer(
            top_k=2,
            num_experts=8,
            hidden_size=128,
            intermediate_size=64,
            quant_config=ct(),
            layer_index=1,
            prefix="model.layers.1.mlp",
            tp_rank=0,
            tp_size=1,
            ep_rank=0,
            ep_size=1,
            routing_config={"precision_policy": "default", **change},
        )


@pytest.mark.parametrize(
    "capability", ["compatible", "kernel_routing", "deferred_finalize"]
)
def test_amd_fp8_auto_respects_registered_capabilities(
    server, amd_platform, capability
):
    from tokenspeed_kernel.registry import KernelRegistry

    server("auto")
    registry = KernelRegistry.get()
    original = registry.get_by_name("triton_fp8_channel_fused_precomputed_moe_apply")
    traits = dict(original.traits)
    if capability == "kernel_routing":
        traits["routing_mode"] = frozenset({"kernel_routing"})
    elif capability == "deferred_finalize":
        traits["supports_deferred_finalize"] = frozenset({True})
    candidate = dataclasses.replace(
        original,
        name="test_compatible_fp8_apply",
        priority=original.priority + 1,
        traits=traits,
    )
    registry.register(candidate, registry.get_impl(original.name))
    try:
        layer = _fp8_layer()
        expected = original.name if capability == "kernel_routing" else candidate.name
        assert layer.experts.plan["apply_kernel_name"] == expected
        assert not layer.kernel_routing
        assert layer.experts.supports_precomputed_topk
        assert not layer.experts.supports_deferred_finalize
    finally:
        registry._unregister(candidate.name)


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_amd_fp8_route_applies_the_scale_once(server, amd_platform, backend):
    server(backend)
    layer = _fp8_layer()
    layer.gate.e_score_correction_bias.data.zero_()
    logits = torch.arange(16, dtype=torch.float32).view(2, 8) / 10
    hidden = torch.zeros(2, 128)
    actual = layer.topk(hidden, logits)
    expected, ids = beam.beam_route_weights(
        logits, layer.gate.e_score_correction_bias, 2, True, 2, 1, None
    )
    assert actual.topk_weights.dtype == torch.float32
    assert torch.equal(actual.topk_ids, ids)
    assert torch.equal(actual.topk_weights, expected * 2.5)
    assert layer.experts.plan.get("output_scale", 1.0) == 1.0


@pytest.mark.parametrize("ep_size", [1, 4])
def test_amd_fp8_moe_partial_takes_one_shared_allreduce(
    server, amd_platform, monkeypatch, ep_size
):
    server("triton")
    config = tiny_config()
    config.moe_intermediate_size = 512
    model = beam.BeamForCausalLM(
        config, Mapping(rank=3, world_size=4, moe_ep_size=ep_size), ct()
    )
    reductions = []

    def allreduce(partial, group):
        reductions.append(tuple(group))
        return partial * 4

    monkeypatch.setattr(comm_module, "all_reduce", allreduce)
    comm = model.model.layers[1].comm_manager
    partial = torch.ones(2, 128, dtype=torch.bfloat16)
    residual = torch.zeros_like(partial)
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=2,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 2, 1),
    )
    assert comm.pre_mlp_comm(partial, ctx) is partial
    output, kept = comm.post_mlp_comm(partial, residual, ctx)
    assert torch.equal(output, partial * 4)
    assert kept is residual
    assert reductions == [(0, 1, 2, 3)]


@pytest.mark.parametrize("backend", ["flashinfer_cutlass"])
def test_amd_quantized_admission_refuses_unqualified_backends(
    server, amd_platform, backend
):
    server(backend)
    from tokenspeed_kernel.selection import NoKernelFoundError

    with pytest.raises(NoKernelFoundError):
        _fp8_layer()


def test_beam_amd_keeps_unfused_boundaries_when_fusion_is_requested(
    server, amd_platform, monkeypatch
):
    server("triton")
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", True)
    model = beam.BeamModel(
        tiny_config(), Mapping(rank=0, world_size=1), ct(), prefix="model"
    )
    assert (
        beam_sandwich_fusion_refusal(model)
        == "sandwich boundary fusion requires NVIDIA kernels"
    )
    assert not any(isinstance(module, BeamSandwichNorm) for module in model.modules())
    assert isinstance(model.layers[0].input_layernorm, beam.BeamRMSNorm)


def test_beam_amd_does_not_admit_other_architectures(server, amd_platform):
    server("triton")
    Platform.override(
        dataclasses.replace(current_platform(), arch_version=ArchVersion(12, 5))
    )
    with pytest.raises(ValueError, match="gfx950 with --moe-backend auto"):
        beam.beam_moe_class(quantized_experts=True)


def _boundary_rmsnorm(x, weight, eps):
    x32 = x.float()
    return (
        x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(x.dtype)


def _boundary_bf16(*shape, seed, scale):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(*shape, generator=generator, dtype=torch.float32) * scale
    ).bfloat16()


@pytest.fixture
def cpu_boundary_rmsnorm(monkeypatch):
    """Only the GPU RMSNorm operation needs a CPU arithmetic substitute."""

    def normalize(x, weight, eps, *, residual, out, enable_pdl):
        assert residual is None
        output = _boundary_rmsnorm(x, weight, eps)
        return output if out is None else out.copy_(output)

    monkeypatch.setattr(beam, "rmsnorm", normalize)


@pytest.mark.parametrize(
    "boundary,with_residual",
    [("first", False), ("first", True), ("later", True), ("final", True)],
)
def test_unfused_beam_boundaries_keep_the_owned_depth_scale_pair(
    server, amd_platform, cpu_boundary_rmsnorm, monkeypatch, boundary, with_residual
):
    server("triton")
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", "unfused")
    model = beam.BeamForCausalLM(
        tiny_config(), Mapping(rank=0, world_size=1), None
    ).model
    layer = model.layers[0 if boundary == "first" else 1]
    norm = model.norm if boundary == "final" else layer.input_layernorm
    norm.weight.data.copy_(_boundary_bf16(128, seed=30, scale=1) + 1)
    hidden = _boundary_bf16(3, 128, seed=31, scale=4)
    residual = _boundary_bf16(3, 128, seed=32, scale=4) if with_residual else None
    if residual is None:
        expected_sum = hidden
    elif norm.residual_fold is None:
        expected_sum = hidden + residual
    else:
        x_scale, residual_scale = norm.residual_fold
        expected_sum = hidden * x_scale + residual * residual_scale
        assert not torch.equal(expected_sum, hidden + residual)
        assert not torch.equal(
            expected_sum,
            (hidden.float() * x_scale + residual.float() * residual_scale).bfloat16(),
        )
    if boundary == "final":
        ctx = ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=3,
            num_extends=0,
            input_num_tokens=3,
            forward_mode=ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(0, 0, 3, 1),
        )
        output, residual_out = layer.comm_manager.final_norm(
            hidden, residual, ctx, norm
        )
    else:
        output, residual_out = layer.comm_manager.input_reduce_norm(hidden, residual)
    assert torch.equal(residual_out, expected_sum)
    assert torch.equal(
        output, _boundary_rmsnorm(expected_sum, norm.weight, norm.variance_epsilon)
    )
    if not with_residual:
        assert residual_out is hidden


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
