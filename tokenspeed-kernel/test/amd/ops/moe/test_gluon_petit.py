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

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_plan
from tokenspeed_kernel.ops.moe.gluon.petit import (
    _DSV4_PROFILE,
    _GPT_OSS_120B_PROFILE,
    _KIMI_K3_PROFILE,
    _GluonPetitState,
    _Profile,
    _validate_layer,
    gluon_petit_mxfp4_megamoe_apply,
    gluon_petit_mxfp4_megamoe_weights,
)
from tokenspeed_kernel.selection import NoKernelFoundError


@pytest.mark.parametrize(
    "profile", [_GPT_OSS_120B_PROFILE, _DSV4_PROFILE, _KIMI_K3_PROFILE]
)
def test_petit_plan_selects_gluon_registration(profile, mi350_platform) -> None:
    with mock.patch(
        "tokenspeed_kernel.selection.current_platform", return_value=mi350_platform
    ):
        plan = moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation=profile.activation,
            routing_mode="precomputed_topk",
            a2a_backend="gluon_petit",
            ep_size=8,
            ispp=profile.logical_intermediate,
            hidden=profile.model_dim,
            swiglu_form="generalized" if profile.has_bias else None,
            activation_clamped=profile.has_bias,
            expert_id_repeats=False,
            internal_activation_dtype="mxfp4",
            with_bias=profile.has_bias,
            fast_math=True,
            combine_order="rank",
            solution="gluon",
        )

    assert plan["solution"] == "gluon"
    assert plan["apply_kernel_name"] == "gluon_petit_mxfp4_megamoe_apply"
    assert plan["weight_preprocessor"] is gluon_petit_mxfp4_megamoe_weights
    assert plan["a2a_backend"] == "gluon_petit"


def test_petit_cannot_replace_explicit_deepep(mi350_platform) -> None:
    with mock.patch(
        "tokenspeed_kernel.selection.current_platform", return_value=mi350_platform
    ), pytest.raises(NoKernelFoundError):
        moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation="swiglu",
            routing_mode="precomputed_topk",
            a2a_backend="deepep",
            ep_size=8,
            ispp=2880,
            hidden=2880,
            swiglu_form="generalized",
            activation_clamped=True,
            expert_id_repeats=False,
            internal_activation_dtype="mxfp4",
            with_bias=True,
            fast_math=True,
            combine_order="rank",
            solution="gluon",
        )


def _gpt_oss_layer() -> SimpleNamespace:
    return SimpleNamespace(
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
        w13_weight_bias=object(),
        w2_weight_bias=object(),
    )


def _dsv4_layer(limit: float | None) -> SimpleNamespace:
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
        w13_weight_bias=None,
        w2_weight_bias=None,
    )


def test_validate_layer_selects_gpt_oss_120b_profile() -> None:
    assert _validate_layer(_gpt_oss_layer()) == _GPT_OSS_120B_PROFILE


def test_validate_layer_preserves_dsv4_activation_contract() -> None:
    assert _validate_layer(_dsv4_layer(None)) == _DSV4_PROFILE
    with pytest.raises(ValueError, match="does not support an activation clamp"):
        _validate_layer(_dsv4_layer(10.0))


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({}, None),
        ({"activation": "silu"}, "SiTU"),
        ({"activation_situ_beta": 3.0}, "SiTU"),
        ({"activation_situ_linear_beta": 24.0}, "SiTU"),
        ({"swiglu_beta": 1.0}, "SiTU"),
        ({"swiglu_arg": SimpleNamespace(alpha=None, limit=7.0)}, "SiTU"),
        ({"w13_input_layout": "interleaved"}, "concatenated"),
        ({"w13_weight_bias": object()}, "bias"),
        ({"w2_weight_bias": object()}, "bias"),
    ],
)
def test_validate_layer_preserves_kimi_k3_activation_contract(overrides, error):
    values = dict(
        num_experts=896,
        top_k=16,
        hidden_size=3584,
        intermediate_size=3072,
        ep_size=8,
        tp_size=1,
        num_local_experts=112,
        activation="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        swiglu_beta=None,
        swiglu_arg=None,
        w13_input_layout="concatenated",
        w13_weight_bias=None,
        w2_weight_bias=None,
    )
    values.update(overrides)
    layer = SimpleNamespace(**values)
    if error:
        with pytest.raises(ValueError, match=error):
            _validate_layer(layer)
    else:
        assert _validate_layer(layer) == _KIMI_K3_PROFILE


def _register_parameter(
    module: torch.nn.Module,
    name: str,
    shape: tuple[int, ...],
) -> None:
    module.register_parameter(
        name,
        torch.nn.Parameter(torch.ones(shape, dtype=torch.uint8), requires_grad=False),
    )


def test_weight_preprocessor_repacks_and_releases_source_parameters() -> None:
    module = torch.nn.Module()
    _register_parameter(module, "w13_weight", (2, 64, 32))
    _register_parameter(module, "w13_weight_scale", (2, 64, 2))
    _register_parameter(module, "w2_weight", (2, 64, 16))
    _register_parameter(module, "w2_weight_scale", (2, 64, 1))
    module.w13_weight_bias = None
    module.w2_weight_bias = None
    module._moe_backend_state = None
    profile = _Profile(
        name="test",
        num_experts=2,
        top_k=1,
        model_dim=64,
        logical_intermediate=32,
        inter_dim=32,
        has_bias=False,
        activation="silu",
    )
    layouts = []

    def repack(
        data: torch.Tensor,
        scales: torch.Tensor | None,
        *,
        layout: object,
        petit_format: bool,
    ):
        assert petit_format
        layouts.append(layout)
        if scales is None:
            return data
        return data, scales

    native_layout = object()
    petit_kernel = SimpleNamespace(
        MoeKernelLayout=SimpleNamespace(native_mxfp4=native_layout),
        repack_moe_kernel_layout=repack,
    )
    with (
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.petit._validate_layer",
            return_value=profile,
        ),
        mock.patch("tokenspeed_kernel.ops.moe.gluon.petit._get_workspace"),
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.petit._import_petit_kernel",
            return_value=petit_kernel,
        ),
        mock.patch("tokenspeed_kernel.ops.moe.gluon.petit.torch.cuda.empty_cache"),
    ):
        gluon_petit_mxfp4_megamoe_weights(plan={}, w=module)

    state = module._moe_backend_state
    assert state.profile == profile
    assert state.w13_weight.shape == (2, 64, 256)
    assert state.w13_scale.shape == (2, 64, 16)
    assert state.w2_weight.shape == (2, 512, 16)
    assert state.w2_scale.shape == (2, 512, 1)
    assert layouts == [native_layout, native_layout]
    for name in (
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ):
        assert getattr(module, name) is None


def test_apply_keeps_zero_token_rank_in_collective() -> None:
    profile = _Profile(
        name="test",
        num_experts=8,
        top_k=1,
        model_dim=64,
        logical_intermediate=32,
        inter_dim=32,
        has_bias=False,
        activation="silu",
    )
    inputs = SimpleNamespace(
        tokens=torch.empty((4, 32), dtype=torch.uint8),
        scales=torch.empty((4, 2), dtype=torch.uint8),
        expert_ids=torch.empty((4, 1), dtype=torch.int32),
        expert_weights=torch.empty((4, 1), dtype=torch.float32),
    )
    config = mock.Mock()
    config.run.side_effect = lambda *args, **kwargs: kwargs["out"]
    workspace = SimpleNamespace(config=config, heap=object(), inputs=inputs)
    layer = SimpleNamespace(
        _moe_backend_state=_GluonPetitState(
            profile=profile,
            w13_weight=torch.empty(0),
            w2_weight=torch.empty(0),
            w13_scale=torch.empty(0),
            w2_scale=torch.empty(0),
            w13_bias=None,
            w2_bias=None,
        )
    )
    overlap = mock.Mock()
    x = torch.empty((0, profile.model_dim), dtype=torch.bfloat16)

    with mock.patch(
        "tokenspeed_kernel.ops.moe.gluon.petit._get_workspace",
        return_value=workspace,
    ):
        output = gluon_petit_mxfp4_megamoe_apply(
            plan={},
            x=x,
            w=layer,
            router_logits=torch.empty((0, 0), dtype=torch.bfloat16),
            topk_weights=torch.empty((0, 1), dtype=torch.float32),
            topk_ids=torch.empty((0, 1), dtype=torch.int32),
            num_tokens_global=1,
            max_num_tokens_per_gpu=1,
            do_finalize=True,
            enable_pdl=False,
            low_latency=None,
            overlap_fn=overlap,
        )

    assert output.shape == (0, profile.model_dim)
    config.quantize.assert_not_called()
    config.run.assert_called_once()
    assert config.run.call_args.args[5] == 0
    overlap.assert_called_once_with()
