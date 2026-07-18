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

import pytest
import torch

mxfp4_preprocess = pytest.importorskip(
    "tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess",
    exc_type=ImportError,
)
gluon_a4w4 = pytest.importorskip(
    "tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950",
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
    assert not hasattr(w13_storage, "gluon_decode_clean_weight")
    assert not hasattr(w13_storage, "gluon_decode_clean_scale")
    assert not hasattr(w2_storage, "gluon_decode_clean_weight")
    assert not hasattr(w2_storage, "gluon_decode_clean_scale")
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


def test_preprocess_gluon_mxfp4_gfx950_keeps_interleaved_runtime_w13_without_clean_copy(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    module = _make_module()
    module.w13_input_layout = "interleaved"

    mxfp4_preprocess.preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)

    assert module.w13_weight_triton_tensor.is_shuffled_for_gluon_dot is True
    assert not hasattr(module.w13_weight_triton_tensor, "gluon_decode_clean_weight")
    assert not hasattr(module.w13_weight_triton_tensor, "gluon_decode_clean_scale")


def test_attach_prefill_aliases_skips_non_gdot128_weights():
    """Regression: the package-prefill alias attach must no-op (not raise) when
    the weights are not gdot128-preshuffled (e.g. gpt-oss), so those models load
    and fall back to the reference path instead of crashing at model load.
    """
    module = torch.nn.Module()
    # Plain 3-D weights WITHOUT the ``is_shuffled_for_gluon_dot`` marker.
    module.w13_weight_triton_tensor = torch.zeros((4, 8, 8), dtype=torch.uint8)
    module.w2_weight_triton_tensor = torch.zeros((4, 8, 8), dtype=torch.uint8)
    scale = torch.zeros((4, 32, 1), dtype=torch.uint8)

    gluon_a4w4.attach_prefill_aliases(module, scale, scale)  # must not raise

    assert not hasattr(module.w13_weight_triton_tensor, "gluon_package_prefill_weight")
    assert not hasattr(module.w2_weight_triton_tensor, "gluon_package_prefill_weight")
