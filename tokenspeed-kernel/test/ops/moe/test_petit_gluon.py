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
from tokenspeed_kernel.ops.moe.gluon.petit import (
    _DSV4_PROFILE,
    _GPT_OSS_120B_PROFILE,
    _Profile,
    _validate_layer,
    petit_gluon_mxfp4_megamoe_apply,
    petit_gluon_mxfp4_megamoe_weights,
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


@pytest.mark.parametrize(
    ("alpha", "limit", "beta"),
    ((1.0, 7.0, 1.0), (1.702, None, 1.0), (1.702, 8.0, 1.0), (1.702, 7.0, 0.0)),
)
def test_validate_layer_rejects_noncanonical_gpt_oss_swiglu(
    alpha: float,
    limit: float | None,
    beta: float,
) -> None:
    layer = _gpt_oss_layer()
    layer.swiglu_arg = SimpleNamespace(alpha=alpha, limit=limit)
    layer.swiglu_beta = beta

    with pytest.raises(ValueError, match="alpha=1.702, limit=7.0, beta=1.0"):
        _validate_layer(layer)


def test_validate_layer_accepts_dsv4_clamp_with_one_warning(caplog) -> None:
    import tokenspeed_kernel.ops.moe.gluon.petit as integration

    with mock.patch.object(integration, "_warned_dsv4_clamp", False):
        assert _validate_layer(_dsv4_layer(10.0)) == _DSV4_PROFILE
        assert _validate_layer(_dsv4_layer(10.0)) == _DSV4_PROFILE

    messages = [record.message for record in caplog.records]
    assert (
        sum("configured activation clamp 10.0 is ignored" in m for m in messages) == 1
    )


def test_validate_layer_rejects_unsupported_geometry() -> None:
    layer = _gpt_oss_layer()
    layer.num_experts = 64

    with pytest.raises(ValueError, match="Unsupported Gluon Petit MegaMoE geometry"):
        _validate_layer(layer)


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
    profile = _Profile(
        name="test",
        num_experts=2,
        top_k=1,
        model_dim=64,
        logical_intermediate=32,
        inter_dim=32,
        has_bias=False,
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
        petit_gluon_mxfp4_megamoe_weights(plan={}, w=module)

    assert module.petit_gluon_profile == profile
    assert module.petit_gluon_w13_weight.shape == (2, 64, 256)
    assert module.petit_gluon_w13_scale.shape == (2, 64, 16)
    assert module.petit_gluon_w2_weight.shape == (2, 512, 16)
    assert module.petit_gluon_w2_scale.shape == (2, 512, 1)
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
        petit_gluon_profile=profile,
        petit_gluon_w13_weight=torch.empty(0),
        petit_gluon_w2_weight=torch.empty(0),
        petit_gluon_w13_scale=torch.empty(0),
        petit_gluon_w2_scale=torch.empty(0),
        petit_gluon_w13_bias=None,
        petit_gluon_w2_bias=None,
    )
    overlap = mock.Mock()
    x = torch.empty((0, profile.model_dim), dtype=torch.bfloat16)

    with mock.patch(
        "tokenspeed_kernel.ops.moe.gluon.petit._get_workspace",
        return_value=workspace,
    ):
        output = petit_gluon_mxfp4_megamoe_apply(
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


def test_apply_rejects_more_than_workspace_capacity() -> None:
    x = torch.empty((1025, 64), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="more than 1024 tokens"):
        petit_gluon_mxfp4_megamoe_apply(
            plan={},
            x=x,
            w=SimpleNamespace(),
            router_logits=torch.empty((1025, 0), dtype=torch.bfloat16),
            topk_weights=torch.empty((1025, 1), dtype=torch.float32),
            topk_ids=torch.empty((1025, 1), dtype=torch.int32),
            num_tokens_global=8200,
            max_num_tokens_per_gpu=None,
            do_finalize=True,
            enable_pdl=False,
            low_latency=None,
            overlap_fn=None,
        )
