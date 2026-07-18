from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch


def _ensure_tokenspeed_triton_importable() -> None:
    try:
        import tokenspeed_triton  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    triton = pytest.importorskip("triton")
    sys.modules["tokenspeed_triton"] = triton
    for submodule in (
        "language",
        "language.core",
        "experimental",
        "experimental.gluon",
        "experimental.gluon.language",
        "experimental.gluon.language.amd",
        "experimental.gluon.language.amd.cdna4",
        "experimental.gluon.language.amd.cdna4.async_copy",
        "experimental.gluon.language.amd.gfx1250",
        "experimental.gluon.language.amd.gfx1250.tdm",
    ):
        sys.modules[f"tokenspeed_triton.{submodule}"] = importlib.import_module(
            f"triton.{submodule}"
        )


_ensure_tokenspeed_triton_importable()

mxfp4_preprocess = pytest.importorskip(
    "tokenspeed_kernel_amd.ops.moe.mxfp4_gfx1250_preprocess",
    exc_type=ImportError,
)


def _make_module() -> torch.nn.Module:
    num_experts = 2
    hidden = 160
    intermediate = 128
    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (num_experts, 2 * intermediate, hidden // 2),
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    module.w2_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (num_experts, hidden, intermediate // 2),
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    module.w13_weight_scale = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (num_experts, 2 * intermediate, hidden // 32),
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    module.w2_weight_scale = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (num_experts, hidden, intermediate // 32),
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    module.w13_weight_bias = torch.nn.Parameter(
        torch.arange(num_experts * 2 * intermediate, dtype=torch.bfloat16).reshape(
            num_experts, 2 * intermediate
        ),
        requires_grad=False,
    )
    module.w2_weight_bias = torch.nn.Parameter(
        torch.ones((num_experts, hidden), dtype=torch.bfloat16),
        requires_grad=False,
    )
    module.w13_input_scale = torch.nn.Parameter(
        torch.tensor([0.5, 0.75], dtype=torch.float32),
        requires_grad=False,
    )
    module.w2_input_scale = torch.nn.Parameter(
        torch.tensor([0.25, 0.625], dtype=torch.float32),
        requires_grad=False,
    )
    return module


def _unswizzle_gfx1250_scale(
    scale: torch.Tensor,
    *,
    k_scale: int,
    n: int,
) -> torch.Tensor:
    b = scale.shape[0]
    align_k_scale = min(4, max(k_scale, 1))
    k_scale_pad = ((k_scale + align_k_scale - 1) // align_k_scale) * align_k_scale
    n_pad = ((n + 127) // 128) * 128
    data = scale.transpose(-1, -2)
    data = data.view(
        b,
        n_pad // 128,
        k_scale_pad // align_k_scale,
        128 // 4,
        4,
        align_k_scale,
    )
    data = data.permute(0, 1, 4, 3, 2, 5)
    data = data.reshape(b, n_pad, k_scale_pad)
    return data.transpose(-1, -2)[..., :k_scale, :n].contiguous()


def _interleave_gate_up_rows(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % tensor.ndim
    gate, up = tensor.split(tensor.shape[dim] // 2, dim=dim)
    shape = list(tensor.shape)
    return torch.stack((gate, up), dim=dim + 1).reshape(shape).contiguous()


def test_preprocess_gluon_mxfp4_gfx1250_mutates_module_state(monkeypatch):
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(1))
    module = _make_module()
    raw_w13_scale = module.w13_weight_scale.detach().clone()
    raw_w2_scale = module.w2_weight_scale.detach().clone()

    mxfp4_preprocess.preprocess_gluon_mxfp4_gfx1250_moe_weights({}, module)

    assert empty_cache_calls == [1]
    assert not hasattr(module, "w13_weight")
    assert not hasattr(module, "w2_weight")
    assert not hasattr(module, "w13_weight_scale")
    assert not hasattr(module, "w2_weight_scale")
    assert module.w13_weight_bias.dtype == torch.float32
    assert module.w2_weight_bias.dtype == torch.float32
    assert module.w13_act_scale.item() == pytest.approx(0.75)
    assert module.w2_act_scale.item() == pytest.approx(0.625)
    assert module.w13_weight_triton_tensor.act_scale is module.w13_act_scale
    assert module.w2_weight_triton_tensor.act_scale is module.w2_act_scale

    assert module.w13_weight_triton_tensor.shape == (2, 80, 256)
    assert module.w2_weight_triton_tensor.shape == (2, 64, 160)
    assert module.w13_weight_triton_tensor.stride(-2) == 1
    assert module.w2_weight_triton_tensor.stride(-2) == 1

    w13_config = module.w13_precision_config
    w2_config = module.w2_precision_config
    assert isinstance(w13_config, mxfp4_preprocess.PrecisionConfig)
    assert isinstance(w2_config, mxfp4_preprocess.PrecisionConfig)
    assert w13_config.b_microblock_size == 32
    assert w2_config.b_microblock_size == 32
    assert w13_config.out_dtype == torch.bfloat16
    assert w2_config.out_dtype == torch.bfloat16

    assert w13_config.b_mx_scale.shape == (2, 1024, 2)
    assert w2_config.b_mx_scale.shape == (2, 512, 2)
    assert w13_config.b_mx_scale.stride(-2) == 1
    assert w2_config.b_mx_scale.stride(-2) == 1
    torch.testing.assert_close(
        _unswizzle_gfx1250_scale(w13_config.b_mx_scale, k_scale=5, n=256),
        _interleave_gate_up_rows(raw_w13_scale, dim=-2).transpose(-2, -1),
    )
    torch.testing.assert_close(
        _unswizzle_gfx1250_scale(w2_config.b_mx_scale, k_scale=4, n=160),
        raw_w2_scale.transpose(-2, -1),
    )


def test_preprocess_gluon_mxfp4_gfx1250_skips_static_scales_for_dynamic_activations(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    module = _make_module()
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)

    mxfp4_preprocess.preprocess_gluon_mxfp4_gfx1250_moe_weights({}, module)

    assert not hasattr(module, "w13_act_scale")
    assert not hasattr(module, "w2_act_scale")
    assert module.w13_weight_triton_tensor.act_scale is None
    assert module.w2_weight_triton_tensor.act_scale is None
    assert module.w13_precision_config.b_mx_scale is not None
    assert module.w2_precision_config.b_mx_scale is not None
