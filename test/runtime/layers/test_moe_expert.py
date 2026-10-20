from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
from tokenspeed.runtime.layers.moe.utils import All2AllBackend, MoeBackend
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")


@pytest.mark.parametrize(
    "backend,dtype,routing_config,error",
    [
        ("flashinfer_cutlass", torch.bfloat16, {}, "requires --moe-backend triton"),
        ("triton", torch.float16, {}, "requires BF16"),
        ("triton", torch.bfloat16, {"routed_scaling_factor": 2.5}, "unscaled"),
    ],
)
def test_moe_bf16_route_precision_rejects_unsupported_contract(
    monkeypatch, backend, dtype, routing_config, error
):
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: MoeBackend(backend))
    policy = {
        "precision_policy": "bf16_route_accum_fp32",
        "output_scale": 2.5,
        "routed_scaling_factor": 1.0,
        **routing_config,
    }
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with pytest.raises(ValueError, match=error):
            MoELayer(
                top_k=2,
                num_experts=4,
                hidden_size=128,
                intermediate_size=32,
                quant_config=None,
                layer_index=0,
                activation="silu",
                routing_mode="precomputed_topk",
                routing_config=policy,
            )
    finally:
        torch.set_default_dtype(original_dtype)


def test_hybrid_moe_dispatches_from_actual_topk_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.plan = {
        "support_routing": True,
        "supports_precomputed_topk": True,
        "supports_deferred_finalize": False,
    }
    calls: list[dict] = []

    def fake_moe_apply(*args, **kwargs):
        calls.append({"router_logits": args[3], **kwargs})
        return args[1]

    monkeypatch.setattr(expert_module, "kernel_moe_apply", fake_moe_apply)

    hidden_states = torch.empty((2, 4), dtype=torch.bfloat16)
    router_logits = torch.empty((2, 8))
    bypassed_output = layer(
        hidden_states,
        BypassedTopKOutput(
            hidden_states,
            router_logits,
            TopKConfig(top_k=2, router_topk="fused"),
            output_scale=torch.ones((2, 1), dtype=torch.float32),
        ),
        num_global_tokens=2,
        max_num_tokens_per_gpu=2,
    )
    layer(
        hidden_states,
        StandardTopKOutput(
            torch.empty((2, 2)),
            torch.empty((2, 2), dtype=torch.int32),
            None,
        ),
        num_global_tokens=2,
        max_num_tokens_per_gpu=2,
    )

    assert bypassed_output.dtype == hidden_states.dtype
    assert calls[0]["router_logits"] is router_logits
    assert calls[1]["router_logits"] is None
    assert "topk_weights" not in calls[0]
    assert "topk_ids" not in calls[0]
    assert calls[1]["topk_weights"].shape == (2, 2)
    assert calls[1]["topk_ids"].shape == (2, 2)


@pytest.mark.parametrize(
    "beta,linear_beta",
    [
        (None, 25.0),
        (0.0, 25.0),
        (4.0, 0.0),
    ],
)
def test_moe_layer_rejects_invalid_situ_parameters(
    beta: float | None,
    linear_beta: float | None,
) -> None:
    with pytest.raises(ValueError, match="beta values must be positive"):
        MoELayer(
            top_k=1,
            num_experts=1,
            hidden_size=32,
            intermediate_size=32,
            quant_config=None,
            layer_index=0,
            activation="situ",
            activation_situ_beta=beta,
            activation_situ_linear_beta=linear_beta,
        )


@pytest.mark.parametrize("backend", ["none", "agrs", "flashinfer"])
@pytest.mark.parametrize("moe_backend", ["auto", "mega_moe"])
def test_moe_layer_builds_ep8_local_expert_partition(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    moe_backend: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_moe_plan(weight_dtype: str, **kwargs) -> dict:
        captured["plan"] = {"weight_dtype": weight_dtype, **kwargs}
        return {
            "solution": kwargs["solution"] or "triton",
            "process_group": kwargs["process_group"],
            "support_routing": False,
            "supports_deferred_finalize": False,
        }

    def fake_create_layer_weights(spec, *args, **kwargs) -> None:
        captured["spec"] = spec

    selected_backend = SimpleNamespace(value=moe_backend)
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: selected_backend)
    ep_group = tuple(range(8))
    process_group = object()
    resolve_group = mock.Mock(return_value=process_group)
    monkeypatch.setattr(
        expert_module.pg_manager, "get_device_process_group", resolve_group
    )
    monkeypatch.setitem(
        expert_module.global_server_args_dict,
        "mapping",
        SimpleNamespace(moe=SimpleNamespace(ep_group=ep_group)),
    )
    monkeypatch.setattr(expert_module, "kernel_moe_plan", fake_moe_plan)
    monkeypatch.setattr(
        expert_module, "create_layer_weights", fake_create_layer_weights
    )

    monkeypatch.setattr(
        expert_module, "get_all2all_backend", lambda: All2AllBackend(backend)
    )
    layer = MoELayer(
        top_k=16,
        num_experts=896,
        hidden_size=3584,
        intermediate_size=3072,
        quant_config=None,
        layer_index=1,
        ep_rank=7,
        ep_size=8,
        activation="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        routing_mode="precomputed_topk",
    )

    assert layer.num_local_experts == 112
    assert layer.ep_rank == 7
    assert layer.ep_size == 8
    assert layer.activation_situ_beta == 4.0
    assert layer.activation_situ_linear_beta == 25.0
    assert captured["spec"].num_local_experts == 112
    assert captured["plan"]["a2a_backend"] == "none"
    assert captured["spec"].a2a_backend == "none"
    assert captured["plan"]["ep_size"] == 8
    assert captured["plan"]["activation"] == "situ"
    assert captured["plan"]["routing_mode"] == "precomputed_topk"

    expected_group = process_group if moe_backend == "mega_moe" else None
    assert captured["plan"]["process_group"] is expected_group
    assert layer.plan["process_group"] is expected_group
    if moe_backend == "mega_moe":
        resolve_group.assert_called_once_with(ep_group)
    else:
        resolve_group.assert_not_called()


def test_moe_layer_rejects_uneven_contiguous_ep_partition() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        MoELayer(
            top_k=2,
            num_experts=10,
            hidden_size=32,
            intermediate_size=32,
            quant_config=None,
            layer_index=0,
            ep_rank=0,
            ep_size=3,
        )


def test_moe_layer_uses_mixed_fp8_block_scale_child_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_moe_plan(weight_dtype: str, **kwargs) -> dict:
        captured["weight_dtype"] = weight_dtype
        captured.update(kwargs)
        return {
            "solution": "flashinfer_trtllm",
            "support_routing": False,
            "supports_deferred_finalize": True,
        }

    trtllm_backend = type("TrtllmBackend", (), {"value": "flashinfer_trtllm"})()
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: trtllm_backend)
    monkeypatch.setattr(expert_module, "kernel_moe_plan", fake_moe_plan)
    quant_config = ModelOptMixedConfig(
        quantized_layers={
            "mtp.layers.0.mlp.experts": "FP8_BLOCK_SCALES",
        }
    )

    layer = MoELayer(
        top_k=2,
        num_experts=4,
        hidden_size=256,
        intermediate_size=640,
        quant_config=quant_config,
        layer_index=0,
        prefix="mtp.layers.0.mlp",
    )

    assert layer.quant_config is quant_config.fp8_block_scales_config
    assert captured["weight_dtype"] == "fp8"
    assert captured["fp8_scale_block_shape"] == (128, 128)
    assert captured["internal_activation_dtype"] == "input"
    assert layer.w13_weight.dtype == torch.float8_e4m3fn
    assert layer.w2_weight.dtype == torch.float8_e4m3fn
    assert layer.w13_weight_scale_inv.dtype == torch.float32
    assert layer.w2_weight_scale_inv.dtype == torch.float32
    assert layer.w13_weight_scale_inv.shape == (4, 10, 2)
    assert layer.w2_weight_scale_inv.shape == (4, 2, 5)


def test_moe_layer_applies_outer_mixed_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_moe_plan(weight_dtype: str, **kwargs) -> dict:
        captured["weight_dtype"] = weight_dtype
        return {
            "solution": "triton",
            "support_routing": False,
            "supports_deferred_finalize": False,
        }

    auto_backend = type("AutoBackend", (), {"value": "auto"})()
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: auto_backend)
    monkeypatch.setattr(expert_module, "kernel_moe_plan", fake_moe_plan)
    monkeypatch.setattr(
        expert_module, "create_layer_weights", lambda *args, **kwargs: None
    )
    quant_config = ModelOptMixedConfig(
        quantized_layers={
            "mtp.layers.0.mlp.experts": "FP8_BLOCK_SCALES",
        },
        exclude_modules=["mtp.layers.0.mlp.experts"],
    )

    layer = MoELayer(
        top_k=2,
        num_experts=4,
        hidden_size=256,
        intermediate_size=640,
        quant_config=quant_config,
        layer_index=0,
        prefix="mtp.layers.0.mlp",
    )

    assert layer.quant_config is quant_config
    assert layer._quant_kind == "unquant"
    assert captured["weight_dtype"] == "unquant"


def test_moe_layer_requests_dynamic_mxfp4_activations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class Mxfp4QuantConfig:
        ignored_layers = None
        exclude_modules = None
        is_w4a8_fp8 = False
        use_dynamic_mxfp4_activations = True

        def get_moe_quant_config(self, prefix: str):
            return self

        def moe_weight_dtype(self, prefix: str) -> str:
            return "mxfp4"

    def fake_moe_plan(weight_dtype: str, **kwargs) -> dict:
        captured.update(kwargs)
        return {
            "solution": "triton",
            "support_routing": False,
            "supports_deferred_finalize": False,
        }

    auto_backend = type("AutoBackend", (), {"value": "auto"})()
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: auto_backend)
    monkeypatch.setattr(expert_module, "kernel_moe_plan", fake_moe_plan)
    monkeypatch.setattr(
        expert_module, "create_layer_weights", lambda *args, **kwargs: None
    )

    MoELayer(
        top_k=2,
        num_experts=4,
        hidden_size=128,
        intermediate_size=128,
        quant_config=Mxfp4QuantConfig(),
        layer_index=0,
    )

    assert captured["internal_activation_dtype"] == "mxfp4"


def test_kernel_routing_rejects_missing_logits(monkeypatch) -> None:
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.plan = {
        "support_routing": True,
        "supports_precomputed_topk": False,
        "supports_deferred_finalize": False,
    }
    apply = mock.Mock(
        side_effect=AssertionError("kernel launched without router logits")
    )
    monkeypatch.setattr(expert_module, "kernel_moe_apply", apply)
    with pytest.raises(ValueError, match="requires router logits"):
        layer(
            torch.ones(1, 4),
            StandardTopKOutput(
                torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32), None
            ),
            num_global_tokens=1,
            max_num_tokens_per_gpu=1,
        )
    apply.assert_not_called()


VENDOR_POLICY = {
    "precision_policy": "bf16_vendor_fp32_route",
    "output_scale": 1.0,
    "routed_scaling_factor": 2.5,
    "n_group": 2,
    "topk_group": 1,
    "normalize_topk_weights": True,
    "correction_bias": torch.zeros(4, dtype=torch.float32),
}
KERNEL_ROUTING_APPLY = "flashinfer_trtllm_unquant_moe_apply"
ROUTED_APPLY = "flashinfer_trtllm_unquant_routed_moe_apply"


@pytest.fixture
def vendor_layer(monkeypatch):
    """Build a vendor-policy MoELayer; the kernel's FP32-bias routing hook
    (``hook.return_value``: whether this GPU builds it) is faked."""
    from tokenspeed_kernel.registry import KernelRegistry

    hook = mock.Mock(return_value=True)
    impl = KernelRegistry.get().get_impl(KERNEL_ROUTING_APPLY)
    platform = current_platform()
    if (
        impl is None
        or not platform.is_nvidia
        or platform.arch_version.major != 10
        or platform.arch_version.minor > 3
    ):
        pytest.skip("FlashInfer's TRT-LLM MoE is registered on NVIDIA SM100 to SM103")
    monkeypatch.setattr(impl, "_tokenspeed_fp32_correction_bias", hook)

    def build(backend="flashinfer_trtllm", **kwargs):
        monkeypatch.setattr(
            expert_module, "get_moe_backend", lambda: MoeBackend(backend)
        )
        combine_order = kwargs.pop("combine_order", "rank")
        monkeypatch.setitem(
            expert_module.global_server_args_dict, "moe_combine_order", combine_order
        )
        layer_kwargs = dict(
            top_k=2,
            num_experts=4,
            hidden_size=128,
            # A multiple of both FlashInfer alignments (64 with the ISPP-64
            # launcher, 128 without), so no padding on any build.
            intermediate_size=384,
            quant_config=None,
            layer_index=0,
            activation="silu",
        )
        layer_kwargs.update(kwargs)
        layer_kwargs["routing_config"] = dict(VENDOR_POLICY)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            return MoELayer(**layer_kwargs)
        finally:
            torch.set_default_dtype(original_dtype)

    build.hook = hook
    return build


def test_vendor_policy_routes_in_the_kernel_when_the_hook_builds_it(vendor_layer):
    layer = vendor_layer()
    plan = layer.plan
    assert plan["apply_kernel_name"] == KERNEL_ROUTING_APPLY
    assert plan["precision_policy"] == "bf16_vendor_fp32_route"
    assert plan["output_scale"] == 1.0
    assert layer.support_routing and not layer.supports_deferred_finalize
    assert layer.topk_output_format.is_bypassed()
    # The routed scale stays in routing_config for the kernel; no padding.
    assert layer.routing_config["routed_scaling_factor"] == 2.5
    assert tuple(layer.w13_weight.shape) == (4, 768, 128)


def test_vendor_policy_runs_on_precomputed_routes_without_the_hook(vendor_layer):
    vendor_layer.hook.return_value = False
    layer = vendor_layer()
    assert layer.plan["apply_kernel_name"] == ROUTED_APPLY
    assert not layer.support_routing and layer.supports_precomputed_topk
    assert not layer.supports_deferred_finalize
    assert layer.plan["fp32_correction_bias"] is True


@pytest.mark.parametrize(
    "build_kwargs,error",
    [
        ({"combine_order": "slot"}, "slot combine"),
        ({"backend": "triton"}, "flashinfer_trtllm"),
        ({"intermediate_size": 288}, "intermediate size per partition"),
    ],
)
def test_vendor_policy_rejects_unsupported_contract(vendor_layer, build_kwargs, error):
    with pytest.raises(ValueError, match=error):
        vendor_layer(**build_kwargs)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="the deferred finalizer is a CUDA library",
)
@torch.inference_mode()
def test_deferred_routes_carry_the_precomputed_weights(monkeypatch) -> None:
    # Triton MXFP4 returns no route weights from a deferred launch; the layer
    # hands back the ones it was given so callers can always finalize.
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: MoeBackend("triton"))
    monkeypatch.setitem(
        expert_module.global_server_args_dict, "moe_mxfp4_fp8_activation", False
    )
    monkeypatch.setitem(
        expert_module.global_server_args_dict, "ep_num_redundant_experts", 0
    )
    torch.manual_seed(0)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            layer = MoELayer(
                top_k=2,
                num_experts=4,
                hidden_size=256,
                intermediate_size=256,
                quant_config=Mxfp4Config(
                    ignored_layers=[], is_checkpoint_mxfp4_serialized=True
                ),
                layer_index=0,
                prefix="mlp",
                activation="swiglu",
                routing_mode="precomputed_topk",
            )
    finally:
        torch.set_default_dtype(original_dtype)
    assert layer.supports_deferred_finalize
    for name, param in layer.named_parameters():
        if name.endswith("_scale"):
            param.data.random_(118, 122)
        else:
            param.data.random_(0, 256)
    layer.process_weights_after_loading(layer)
    x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    routes = StandardTopKOutput(
        torch.rand(5, 2, device="cuda"),
        torch.randint(0, 4, (5, 2), device="cuda", dtype=torch.int32),
        None,
    )
    expected = layer(
        x, routes, num_global_tokens=5, max_num_tokens_per_gpu=5, do_finalize=True
    )
    gemm2_out, weights, expanded_idx = layer(
        x, routes, num_global_tokens=5, max_num_tokens_per_gpu=5, do_finalize=False
    )
    assert weights is routes.topk_weights
    actual = moe_finalize_fuse_shared(
        gemm2_out, expanded_idx, weights, None, top_k=2, hidden_dim=256
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("nvidia", [True, False])
def test_only_nvidia_defers_to_the_cuda_finalizer(monkeypatch, nvidia) -> None:
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.plan = {"supports_deferred_finalize": True}
    monkeypatch.setattr(
        expert_module, "current_platform", lambda: SimpleNamespace(is_nvidia=nvidia)
    )
    assert layer.supports_deferred_finalize
    assert layer.supports_deferred_cuda_finalize == nvidia
