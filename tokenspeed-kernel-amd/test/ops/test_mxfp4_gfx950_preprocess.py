from __future__ import annotations

import pytest
import torch

mxfp4_preprocess = pytest.importorskip(
    "tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess",
    exc_type=ImportError,
)


def _make_module() -> torch.nn.Module:
    num_experts = 2
    hidden = 64
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
        torch.ones((num_experts, 2 * intermediate), dtype=torch.bfloat16),
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


def test_preprocess_gluon_mxfp4_gfx950_mutates_module_state(monkeypatch):
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(1))
    module = _make_module()

    mxfp4_preprocess.preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)

    assert empty_cache_calls == [1]
    assert not hasattr(module, "w13_weight")
    assert not hasattr(module, "w2_weight")
    assert module.w13_weight_bias.dtype == torch.float32
    assert module.w2_weight_bias.dtype == torch.float32
    assert module.w13_act_scale.item() == pytest.approx(0.75)
    assert module.w2_act_scale.item() == pytest.approx(0.625)

    w13_storage = module.w13_weight_triton_tensor
    w2_storage = module.w2_weight_triton_tensor
    assert w13_storage.dtype == torch.uint8
    assert w2_storage.dtype == torch.uint8
    assert module.w13_weight_triton_tensor.shape == (2, 128, 256)
    assert module.w2_weight_triton_tensor.shape == (2, 128, 128)
    assert module._w2_logical_n == 64
    assert module.w2_weight_bias.shape == (2, 64)

    assert w13_storage.is_shuffled_for_gluon_dot is True
    assert w2_storage.is_shuffled_for_gluon_dot is True
    assert w13_storage.original_k_pk == 32
    assert w2_storage.original_k_pk == 64
    assert w13_storage.gluon_dot_block_k_pk == 128
    assert w2_storage.gluon_dot_block_k_pk == 128
    assert w13_storage.gluon_dot_block_n == 128
    assert w2_storage.gluon_dot_block_n == 128
    assert not hasattr(w13_storage, "_gluon_shuffled")
    assert not hasattr(w2_storage, "_gluon_shuffled")
    assert w2_storage.original_n == 64
    assert module.w2_weight_triton_tensor.original_n == 64

    w13_config = module.w13_precision_config
    w2_config = module.w2_precision_config
    assert isinstance(w13_config, mxfp4_preprocess.PrecisionConfig)
    assert isinstance(w2_config, mxfp4_preprocess.PrecisionConfig)
    assert w13_config.flex_ctx.lhs_data.dtype == torch.float8_e4m3fn
    assert w2_config.flex_ctx.lhs_data.dtype == torch.float8_e4m3fn
    assert w13_config.flex_ctx.lhs_data.scale is module.w13_act_scale
    assert w2_config.flex_ctx.lhs_data.scale is module.w2_act_scale
    assert w13_config.b_microblock_size == 32
    assert w2_config.b_microblock_size == 32
    assert w13_config.out_dtype == torch.bfloat16
    assert w2_config.out_dtype == torch.bfloat16
    assert w13_config.b_mx_scale.dtype == torch.uint8
    assert w2_config.b_mx_scale.dtype == torch.uint8
    assert w13_config.b_mx_scale.shape == (2, 256, 8)
    assert w2_config.b_mx_scale.shape == (2, 256, 4)
    assert w13_config.b_mx_scale.stride(-2) == 1
    assert w2_config.b_mx_scale.stride(-2) == 1


def test_preprocess_gluon_mxfp4_gfx950_can_disable_preshuffle(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    module = _make_module()

    mxfp4_preprocess.preprocess_gluon_mxfp4_gfx950_moe_weights(
        {}, module, preshuffle=False
    )

    w13_storage = module.w13_weight_triton_tensor
    w2_storage = module.w2_weight_triton_tensor
    assert module.w13_weight_triton_tensor.shape == (2, 32, 256)
    assert module.w2_weight_triton_tensor.shape == (2, 64, 128)
    assert module.w13_weight_triton_tensor.stride(-2) == 1
    assert module.w2_weight_triton_tensor.stride(-2) == 1
    assert module._w2_logical_n == 64
    assert module.w2_weight_triton_tensor.original_n == 64
    assert w2_storage.original_n == 64
    assert module.w2_weight_bias.shape == (2, 64)
    assert not hasattr(w13_storage, "_gluon_shuffled")
    assert not hasattr(w2_storage, "_gluon_shuffled")
    assert not hasattr(module.w13_weight_triton_tensor, "_gluon_shuffled")
    assert not hasattr(module.w2_weight_triton_tensor, "_gluon_shuffled")
