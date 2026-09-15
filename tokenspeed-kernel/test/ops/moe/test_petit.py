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

import runpy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.moe.petit import (
    _import_petit_kernel,
    _Profile,
    _validate_layer,
    petit_mxfp4_megamoe_weights,
)


def _register_parameter(
    module: torch.nn.Module,
    name: str,
    shape: tuple[int, ...],
) -> None:
    module.register_parameter(
        name,
        torch.nn.Parameter(torch.ones(shape, dtype=torch.uint8), requires_grad=False),
    )


def test_validate_layer_selects_gpt_oss_profile() -> None:
    module = SimpleNamespace(
        num_experts=128,
        top_k=4,
        hidden_size=2880,
        intermediate_size=2880,
        ep_size=8,
        tp_size=1,
        num_local_experts=16,
        activation="swiglu",
        swiglu_beta=1.0,
        swiglu_arg=SimpleNamespace(alpha=1.702, limit=7.0),
        w13_input_layout="interleaved",
    )

    profile = _validate_layer(module)

    assert profile == _Profile(
        num_experts=128,
        top_k=4,
        model_dim=2880,
        logical_intermediate=2880,
        inter_dim=3072,
        has_bias=True,
    )


@pytest.mark.parametrize(
    "alpha,limit,beta",
    [
        (1.0, 7.0, 1.0),
        (1.702, None, 1.0),
        (1.702, 8.0, 1.0),
        (1.702, 7.0, 0.0),
    ],
)
def test_validate_layer_rejects_noncanonical_gpt_oss_swiglu(
    alpha: float,
    limit: float | None,
    beta: float,
) -> None:
    module = SimpleNamespace(
        num_experts=128,
        top_k=4,
        hidden_size=2880,
        intermediate_size=2880,
        ep_size=8,
        tp_size=1,
        num_local_experts=16,
        activation="swiglu",
        swiglu_beta=beta,
        swiglu_arg=SimpleNamespace(alpha=alpha, limit=limit),
        w13_input_layout="interleaved",
    )

    with pytest.raises(
        ValueError,
        match="alpha=1.702, limit=7.0, beta=1.0",
    ):
        _validate_layer(module)


def test_validate_layer_rejects_unsupported_geometry() -> None:
    module = SimpleNamespace(
        num_experts=64,
        top_k=4,
        hidden_size=2880,
        intermediate_size=2880,
    )

    with pytest.raises(ValueError, match="Unsupported Petit MegaMoE geometry"):
        _validate_layer(module)


def test_weight_preprocessor_repacks_and_releases_source_parameters() -> None:
    module = torch.nn.Module()
    _register_parameter(module, "w13_weight", (2, 64, 32))
    _register_parameter(module, "w13_weight_scale", (2, 64, 2))
    _register_parameter(module, "w2_weight", (2, 64, 16))
    _register_parameter(module, "w2_weight_scale", (2, 64, 1))
    profile = _Profile(
        num_experts=2,
        top_k=1,
        model_dim=64,
        logical_intermediate=32,
        inter_dim=32,
        has_bias=False,
    )
    layouts = []

    def repack(weight: torch.Tensor, scale: torch.Tensor, *, layout: object):
        layouts.append(layout)
        return weight, scale

    native_layout = object()
    petit_kernel = SimpleNamespace(
        MoeKernelLayout=SimpleNamespace(native_mxfp4=native_layout),
        repack_moe_kernel_layout=repack,
    )

    with (
        mock.patch(
            "tokenspeed_kernel.ops.moe.petit._validate_layer",
            return_value=profile,
        ),
        mock.patch("tokenspeed_kernel.ops.moe.petit._get_workspace"),
        mock.patch(
            "tokenspeed_kernel.ops.moe.petit._import_petit_kernel",
            return_value=petit_kernel,
        ),
    ):
        petit_mxfp4_megamoe_weights({}, module)

    assert module.petit_w13_weight.shape == (2, 64, 256)
    assert module.petit_w13_scale.shape == (2, 64, 16)
    assert module.petit_w2_weight.shape == (2, 512, 16)
    assert module.petit_w2_scale.shape == (2, 512, 1)
    assert layouts == [native_layout, native_layout]
    for name in (
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ):
        assert getattr(module, name) is None


def _v4_layer(limit: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=384,
        top_k=6,
        hidden_size=7168,
        intermediate_size=3072,
        ep_size=8,
        tp_size=1,
        num_local_experts=48,
        activation="swiglu",
        swiglu_beta=None,
        swiglu_arg=SimpleNamespace(alpha=None, limit=limit),
        w13_input_layout="concatenated",
    )


def test_v4_profile_requires_clamp10() -> None:
    assert _validate_layer(_v4_layer(10.0)).num_experts == 384
    for limit in (None, 7.0, 11.0):
        with pytest.raises(ValueError, match="requires activation clamp 10.0"):
            _validate_layer(_v4_layer(limit))


def test_v4_workspace_selects_clamped_silu() -> None:
    from tokenspeed_kernel.ops.moe.petit import _get_workspace

    config = mock.Mock(
        return_value=SimpleNamespace(input_views=lambda heap, rows: (heap, rows))
    )
    kernel = SimpleNamespace(
        MegaMoeConfig=config,
        MegaMoeActivation=SimpleNamespace(mxfp4=object()),
        MegaMoeActivationFunction=SimpleNamespace(silu_clamp10=object()),
        MegaMoeStages=SimpleNamespace(two_stage=object()),
        create_vmm_symmetric_heap=mock.Mock(return_value=object()),
    )
    with (
        mock.patch("tokenspeed_kernel.ops.moe.petit._workspace_cache", {}),
        mock.patch(
            "tokenspeed_kernel.ops.moe.petit.dist.is_initialized", return_value=True
        ),
        mock.patch(
            "tokenspeed_kernel.ops.moe.petit.dist.get_world_size", return_value=8
        ),
        mock.patch(
            "tokenspeed_kernel.ops.moe.petit._import_petit_kernel", return_value=kernel
        ),
    ):
        workspace = _get_workspace(
            torch.device("cuda:0"), _validate_layer(_v4_layer(10.0))
        )
    assert workspace.inputs[1] == 1024
    config.assert_called_once_with(
        world_size=8,
        num_experts=384,
        topk=6,
        model_dim=7168,
        activation=kernel.MegaMoeActivation.mxfp4,
        activation_function=kernel.MegaMoeActivationFunction.silu_clamp10,
        stages=kernel.MegaMoeStages.two_stage,
        inter_dim=3072,
        has_bias=False,
    )
    kernel.create_vmm_symmetric_heap.assert_called_once_with(8)


@pytest.mark.parametrize("num_experts", [256, 384])
def test_existing_deepseek_profiles_still_require_unclamped_silu(
    num_experts: int,
) -> None:
    layer = _v4_layer(None)
    layer.num_experts = num_experts
    layer.num_local_experts = num_experts // 8
    layer.top_k = 8
    layer.intermediate_size = 2048
    assert _validate_layer(layer).num_experts == num_experts
    layer.swiglu_arg.limit = 10.0
    with pytest.raises(ValueError, match="does not support activation clamps"):
        _validate_layer(layer)


def test_import_petit_kernel_returns_available_module() -> None:
    kernel = ModuleType("petit_kernel")
    with mock.patch.dict("sys.modules", {"petit_kernel": kernel}):
        assert _import_petit_kernel() is kernel


def test_petit_integration_import_is_lazy() -> None:
    import tokenspeed_kernel.ops.moe.petit as integration

    with (
        mock.patch.dict("sys.modules", {"petit_kernel": None}),
        mock.patch(
            "tokenspeed_kernel.registry.register_kernel",
            side_effect=lambda family, operation, **kwargs: lambda function: function,
        ),
    ):
        namespace = runpy.run_path(str(Path(integration.__file__)))
        with pytest.raises(
            RuntimeError, match="petit_kernel is not installed"
        ) as error:
            namespace["_import_petit_kernel"]()
        assert isinstance(error.value.__cause__, ImportError)
