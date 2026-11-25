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

import logging
import re
from functools import partial
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_plan
from tokenspeed_kernel.ops.moe.gluon import (
    _DSV4_PROFILE,
    _DSV41_FLASH_DSPARK_PROFILE,
    _DSV41_FLASH_PROFILE,
    _GPT_OSS_120B_PROFILE,
    _KIMI_K3_PROFILE,
    _MUSE_SPARK_PROFILE,
    _compile_variants,
    _compiled_variant_keys,
    _get_workspace,
    _GluonPetitState,
    _import_petit_kernel,
    _megamoe_config,
    _Profile,
    _profile_largest_max_tokens_per_rank,
    _validate_layer,
    _workspace_cache,
    gluon_petit_largest_max_tokens_per_rank,
    gluon_petit_max_tokens_per_rank,
    gluon_petit_mxfp4_megamoe_apply,
    gluon_petit_mxfp4_megamoe_weights,
)
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.selection import NoKernelFoundError


@pytest.mark.parametrize(
    "profile",
    [_GPT_OSS_120B_PROFILE, _DSV4_PROFILE, _KIMI_K3_PROFILE, _MUSE_SPARK_PROFILE],
)
def test_petit_plan_selects_gluon_registration(profile, mi350_platform) -> None:
    with mock.patch(
        "tokenspeed_kernel.selection.current_platform", return_value=mi350_platform
    ):
        plan = moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation=profile.activation,
            requires_deferred_finalize=True,
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
            features=frozenset(),
            solution="gluon",
        )

    assert plan["solution"] == "gluon"
    assert plan["apply_kernel_name"] == "gluon_petit_mxfp4_megamoe_apply"
    assert plan["weight_preprocessor"] is gluon_petit_mxfp4_megamoe_weights
    assert plan["a2a_backend"] == "gluon_petit"
    # Per-route consumers take the unsummed routes.
    assert plan["supports_deferred_finalize"]
    # C1 in the serving step: the router stores its weights in the plan's
    # dtype, which must be Petit's FP32 so Petit reads them in place (no cast).
    assert plan["topk_weights_dtype"] == torch.float32


def test_petit_cannot_replace_explicit_deepep(mi350_platform) -> None:
    with (
        mock.patch(
            "tokenspeed_kernel.selection.current_platform", return_value=mi350_platform
        ),
        pytest.raises(NoKernelFoundError),
    ):
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
            features=frozenset(),
            solution="gluon",
        )


@pytest.mark.parametrize("ep_size,selected", [(4, True), (8, True), (2, False)])
def test_petit_plan_ep_sizes(ep_size, selected, mi350_platform) -> None:
    profile = _DSV41_FLASH_PROFILE
    with mock.patch(
        "tokenspeed_kernel.selection.current_platform", return_value=mi350_platform
    ):
        try:
            plan = moe_plan(
                "mxfp4",
                input_dtype=torch.bfloat16,
                activation="swiglu",
                routing_mode="precomputed_topk",
                a2a_backend="gluon_petit",
                ep_size=ep_size,
                ispp=profile.logical_intermediate,
                hidden=profile.model_dim,
                swiglu_form="standard",
                activation_clamped=True,
                expert_id_repeats=False,
                internal_activation_dtype="mxfp4",
                with_bias=False,
                fast_math=True,
                combine_order="rank",
                features=frozenset(),
                solution="gluon",
            )
        except NoKernelFoundError:
            plan = None
    assert (plan is not None) == selected


def _gpt_oss_layer(ep_size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=128,
        top_k=4,
        hidden_size=2880,
        intermediate_size=2880,
        ep_size=ep_size,
        tp_size=1,
        num_local_experts=128 // ep_size,
        activation="swiglu",
        swiglu_beta=1.0,
        swiglu_arg=SimpleNamespace(alpha=1.702, limit=7.0),
        w13_input_layout="interleaved",
        w13_weight_bias=object(),
        w2_weight_bias=object(),
    )


def _dsv4_layer(limit: float | None, ep_size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=384,
        top_k=6,
        hidden_size=7168,
        intermediate_size=3072,
        ep_size=ep_size,
        tp_size=1,
        num_local_experts=384 // ep_size,
        activation="swiglu",
        swiglu_beta=None,
        swiglu_arg=SimpleNamespace(alpha=None, limit=limit),
        w13_input_layout="concatenated",
        w13_weight_bias=None,
        w2_weight_bias=None,
    )


def _muse_spark_layer(ep_size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=256,
        top_k=8,
        hidden_size=4096,
        intermediate_size=4096,
        ep_size=ep_size,
        tp_size=1,
        num_local_experts=256 // ep_size,
        activation="silu",
        swiglu_beta=None,
        swiglu_arg=None,
        w13_input_layout="concatenated",
        w13_weight_bias=None,
        w2_weight_bias=None,
    )


def test_validate_layer_selects_muse_spark_profile() -> None:
    assert _validate_layer(_muse_spark_layer()) == _MUSE_SPARK_PROFILE


def test_validate_layer_selects_gpt_oss_120b_profile() -> None:
    assert _validate_layer(_gpt_oss_layer()) == _GPT_OSS_120B_PROFILE


def test_validate_layer_preserves_dsv4_activation_contract() -> None:
    assert _validate_layer(_dsv4_layer(None)) == _DSV4_PROFILE
    with pytest.raises(ValueError, match="does not support an activation clamp"):
        _validate_layer(_dsv4_layer(10.0))


def _dsv41_layer(profile: _Profile, ep_size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=profile.num_experts,
        top_k=profile.top_k,
        hidden_size=profile.model_dim,
        intermediate_size=profile.logical_intermediate,
        ep_size=ep_size,
        tp_size=1,
        num_local_experts=profile.num_experts // ep_size,
        activation="swiglu",
        swiglu_beta=None,
        swiglu_arg=SimpleNamespace(alpha=None, limit=10.0),
        w13_input_layout="concatenated",
        w13_weight_bias=None,
        w2_weight_bias=None,
    )


def _kimi_k3_layer(ep_size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=896,
        top_k=16,
        hidden_size=3584,
        intermediate_size=3072,
        ep_size=ep_size,
        tp_size=1,
        num_local_experts=896 // ep_size,
        activation="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        swiglu_beta=None,
        swiglu_arg=None,
        w13_input_layout="concatenated",
        w13_weight_bias=None,
        w2_weight_bias=None,
    )


@pytest.mark.parametrize(
    "profile", [_DSV41_FLASH_PROFILE, _DSV41_FLASH_DSPARK_PROFILE], ids=lambda p: p.name
)
@pytest.mark.parametrize("ep_size", [4, 8])
def test_validate_layer_accepts_dsv41_at_ep4_and_ep8(profile, ep_size) -> None:
    layer = _dsv41_layer(profile, ep_size)
    assert _validate_layer(layer) == profile


@pytest.mark.parametrize(
    "build,ep_size,error",
    [
        (_gpt_oss_layer, 4, "supports EP8 for gpt_oss_120b; got EP4"),
        (partial(_dsv4_layer, None), 4, "supports EP8 for dsv4; got EP4"),
        (_kimi_k3_layer, 4, "supports EP8 for kimi_k3; got EP4"),
        (_muse_spark_layer, 4, "supports EP8 for muse_spark; got EP4"),
        (
            partial(_dsv41_layer, _DSV41_FLASH_PROFILE),
            2,
            "supports EP4/8 for dsv41_flash; got EP2",
        ),
    ],
    ids=[
        "gpt_oss_120b-ep4",
        "dsv4-ep4",
        "kimi_k3-ep4",
        "muse_spark-ep4",
        "dsv41_flash-ep2",
    ],
)
def test_validate_layer_rejects_ep_sizes_outside_the_profile(
    build, ep_size, error
) -> None:
    # Each layer is valid at the default EP8, so only the EP size can trigger the error.
    _validate_layer(build())
    with pytest.raises(ValueError, match=error):
        _validate_layer(build(ep_size=ep_size))


@pytest.mark.parametrize(
    "profile,world_size,error",
    [
        (_DSV4_PROFILE, 4, "world size of 8 for dsv4; got 4"),
        (_DSV41_FLASH_PROFILE, 2, "world size of 4/8 for dsv41_flash; got 2"),
    ],
    ids=["dsv4-ws4", "dsv41_flash-ws2"],
)
def test_workspace_rejects_world_size_outside_the_profile(
    profile, world_size, error
) -> None:
    # The workspace is sized from the process group, which _validate_layer never
    # sees; only the missing process group is faked.
    with (
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.dist.is_initialized", return_value=True
        ),
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.dist.get_world_size",
            return_value=world_size,
        ),
        pytest.raises(RuntimeError, match=error),
    ):
        _get_workspace(torch.device("cuda", 0), profile)


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
    layer = SimpleNamespace(**{**vars(_kimi_k3_layer()), **overrides})
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
            "tokenspeed_kernel.ops.moe.gluon._validate_layer",
            return_value=profile,
        ),
        mock.patch("tokenspeed_kernel.ops.moe.gluon._get_workspace"),
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon._import_petit_kernel",
            return_value=petit_kernel,
        ),
        mock.patch("tokenspeed_kernel.ops.moe.gluon.torch.cuda.empty_cache"),
    ):
        gluon_petit_mxfp4_megamoe_weights(plan={}, w=module)

    state = module._moe_backend_state
    assert state.profile == profile
    # A plan without a server capacity keeps Petit's default 1024 rows.
    assert state.max_tokens_per_rank == 1024
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
        "tokenspeed_kernel.ops.moe.gluon._get_workspace",
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


@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize(
    "id_dtype,weight_dtype",
    [(torch.int32, torch.float32), (torch.int64, torch.bfloat16)],
)
@pytest.mark.parametrize("do_finalize", (True, False))
def test_apply_fuse_prep_reads_router_buffers_in_place(
    fused, id_dtype, weight_dtype, do_finalize
) -> None:
    """C1: with TOKENSPEED_PETIT_FUSE_PREP Petit gets the router's buffers.

    Off, the ids and weights are copied into the heap views (the old path).
    On, int32 ids and FP32 weights are passed through untouched (no copy
    node); other dtypes are cast once; the heap views are not written.
    """
    from tokenspeed_kernel.ops.moe import gluon as petit

    profile = _Profile(
        name="test",
        num_experts=8,
        top_k=2,
        model_dim=64,
        logical_intermediate=32,
        inter_dim=32,
        has_bias=False,
        activation="silu",
    )
    heap = SimpleNamespace(
        tokens=torch.empty((4, 32), dtype=torch.uint8),
        scales=torch.empty((4, 2), dtype=torch.uint8),
        expert_ids=torch.full((4, 2), -1, dtype=torch.int32),
        expert_weights=torch.full((4, 2), -1.0, dtype=torch.float32),
    )
    config = mock.Mock()
    config.run.side_effect = lambda *args, **kwargs: kwargs["out"]
    config.run_routes.side_effect = lambda *args, **kwargs: torch.empty(0)
    workspace = SimpleNamespace(config=config, heap=object(), inputs=heap)
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
    x = torch.zeros((3, profile.model_dim), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 5], [7, 2], [3, 4]], dtype=id_dtype)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.5, 0.5], [0.125, 0.875]], dtype=weight_dtype
    )
    with (
        mock.patch.object(petit, "_PETIT_FUSE_PREP", fused),
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon._get_workspace", return_value=workspace
        ),
    ):
        gluon_petit_mxfp4_megamoe_apply(
            plan={},
            x=x,
            w=layer,
            router_logits=torch.empty((3, 0), dtype=torch.bfloat16),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_tokens_global=3,
            max_num_tokens_per_gpu=3,
            do_finalize=do_finalize,
            enable_pdl=False,
            low_latency=None,
            overlap_fn=None,
        )
    launch = config.run if do_finalize else config.run_routes
    launch.assert_called_once()
    inputs = launch.call_args.kwargs["inputs"]
    config.quantize.assert_called_once()
    assert inputs.tokens.data_ptr() == heap.tokens.data_ptr()
    assert inputs.expert_ids.dtype == torch.int32
    assert inputs.expert_weights.dtype == torch.float32
    assert inputs.expert_ids.is_contiguous() and inputs.expert_weights.is_contiguous()
    torch.testing.assert_close(inputs.expert_ids, topk_ids.to(torch.int32))
    expected_weights = (
        topk_weights.float() if do_finalize else torch.ones((3, 2), dtype=torch.float32)
    )
    torch.testing.assert_close(inputs.expert_weights, expected_weights)
    if fused:
        # The heap's id rows are never written; Petit reads the router's.
        assert (heap.expert_ids == -1).all()
        assert (inputs.expert_ids.data_ptr() == topk_ids.data_ptr()) == (
            id_dtype == torch.int32
        )
        if do_finalize:
            assert (heap.expert_weights == -1).all()
            assert (inputs.expert_weights.data_ptr() == topk_weights.data_ptr()) == (
                weight_dtype == torch.float32
            )
    else:
        assert inputs.expert_ids.data_ptr() == heap.expert_ids.data_ptr()
        assert inputs.expert_weights.data_ptr() == heap.expert_weights.data_ptr()
        assert (heap.expert_ids[3] == -1).all()


@pytest.mark.parametrize(
    "warming,capturing,compiles", [(True, False, 1), (False, False, 0), (True, True, 0)]
)
def test_apply_compiles_every_variant_once_under_startup_warmup(
    warming, capturing, compiles
) -> None:
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
    config = mock.Mock(world_size=8)
    config.run.side_effect = lambda *args, **kwargs: kwargs["out"]
    config.compile_variants.return_value = (0, 1)
    heap = object()
    workspace = SimpleNamespace(config=config, heap=heap, inputs=inputs)
    state = _GluonPetitState(
        profile=profile,
        w13_weight=torch.empty(0),
        w2_weight=torch.empty(0),
        w13_scale=torch.empty(0),
        w2_scale=torch.empty(0),
        w13_bias=None,
        w2_bias=None,
    )
    layer = SimpleNamespace(_moe_backend_state=state)
    x = torch.empty((0, profile.model_dim), dtype=torch.bfloat16)

    with (
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon._get_workspace", return_value=workspace
        ),
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon._workspace_key",
            return_value=(0, profile, 1024),
        ),
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
        kernel_warmup(enabled=warming),
    ):
        _compiled_variant_keys.clear()
        # Two layers of one profile: the variants compile once per workspace.
        for _ in range(2):
            gluon_petit_mxfp4_megamoe_apply(
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
                overlap_fn=None,
            )
    _compiled_variant_keys.clear()

    assert config.compile_variants.call_count == compiles
    if compiles:
        call = config.compile_variants.call_args
        assert call.args == (
            heap,
            state.w13_weight,
            state.w2_weight,
            state.w13_scale,
            state.w2_scale,
        )
        assert call.kwargs == {
            "w13_bias": None,
            "w2_bias": None,
            "inputs": inputs,
        }
    # The serving launches are unchanged.
    assert config.run.call_count == 2


# One rank's VMM heap (reserved, physical) bytes at Petit's default 1024-row
# capacity, as TokenSpeed a0792eef builds it; a raised capacity must not move
# them.
_DEFAULT_CAPACITY_HEAP_BYTES = {
    (_DSV41_FLASH_PROFILE.name, 4): (595546112, 180875264),
    (_DSV41_FLASH_DSPARK_PROFILE.name, 4): (294031360, 90726400),
    (_DSV41_FLASH_PROFILE.name, 8): (1686024192, 276680704),
    (_DSV41_FLASH_DSPARK_PROFILE.name, 8): (838574080, 138260480),
}


@pytest.mark.parametrize("profile", [_DSV41_FLASH_PROFILE, _DSV41_FLASH_DSPARK_PROFILE])
@pytest.mark.parametrize("world_size", [4, 8])
def test_default_capacity_keeps_solution_and_heap(profile, world_size) -> None:
    petit_kernel = _import_petit_kernel()
    default = _megamoe_config(petit_kernel, profile, world_size, 1024)
    # The capacity field (solution-id bits 52-59) is zero at 1024 rows, so the
    # solution id, kernel constants and caches are today's.
    assert default._solution_id >> 52 == 0
    assert default.max_tokens_per_rank == 1024
    assert (
        default.heap_bytes() == _DEFAULT_CAPACITY_HEAP_BYTES[(profile.name, world_size)]
    )
    assert gluon_petit_max_tokens_per_rank(world_size, 1) == 1024
    assert gluon_petit_max_tokens_per_rank(world_size, 1024) == 1024


def test_largest_capacity_per_world_size() -> None:
    # EP4: the DSv4.1 Flash target's heap reaches 2**32 above 7808 rows; the
    # DSpark draft fits every encodable capacity. EP8: the rank-slot region
    # grows with ranks squared, so the target reaches 2**32 above 2624 rows and
    # the draft above 5280; their server's limit is the smaller. Muse Spark's
    # 2368 rows at EP8 do not lower the server-wide limit, the largest over
    # models; its workspace enforces its own.
    assert gluon_petit_largest_max_tokens_per_rank(4) == 7808
    assert gluon_petit_largest_max_tokens_per_rank(8) == 2624
    assert (
        _import_petit_kernel().MEGA_MOE_LARGEST_ENCODED_MAX_TOKENS_PER_RANK == 9184
    ), "the EP4 draft's largest is Petit's largest encodable capacity"


@pytest.mark.parametrize(
    "profile,world_size,largest",
    [
        (_DSV41_FLASH_PROFILE, 4, 7808),
        (_DSV41_FLASH_DSPARK_PROFILE, 4, 9184),
        (_DSV41_FLASH_PROFILE, 8, 2624),
        (_DSV41_FLASH_DSPARK_PROFILE, 8, 5280),
        # Muse Spark's heap reaches 2**32 above 2368 rows at EP8.
        (_MUSE_SPARK_PROFILE, 8, 2368),
        # Profiles that do not raise their capacity at EP8 keep 1024 rows.
        (_GPT_OSS_120B_PROFILE, 8, 1024),
        (_DSV4_PROFILE, 8, 1024),
        (_KIMI_K3_PROFILE, 8, 1024),
        # Profiles not served at EP4 keep 1024 rows there.
        (_GPT_OSS_120B_PROFILE, 4, 1024),
        (_DSV4_PROFILE, 4, 1024),
        (_KIMI_K3_PROFILE, 4, 1024),
        (_MUSE_SPARK_PROFILE, 4, 1024),
    ],
)
def test_largest_capacity_per_profile(profile, world_size, largest) -> None:
    assert _profile_largest_max_tokens_per_rank(profile, world_size) == largest
    assert gluon_petit_max_tokens_per_rank(world_size, largest, profile) == largest
    with pytest.raises(
        ValueError,
        match=f"at most {largest} tokens per rank at world size {world_size} "
        f"for {profile.name}",
    ):
        gluon_petit_max_tokens_per_rank(world_size, largest + 1, profile)
    petit_kernel = _import_petit_kernel()
    if 1024 < largest < petit_kernel.MEGA_MOE_LARGEST_ENCODED_MAX_TOKENS_PER_RANK:
        # The limit is where the heap reaches 32-bit offsets.
        reserved, _ = _megamoe_config(
            petit_kernel, profile, world_size, largest
        ).heap_bytes()
        assert reserved < 1 << 32
        with pytest.raises(ValueError, match="32-bit offsets"):
            _megamoe_config(petit_kernel, profile, world_size, largest + 32)


@pytest.mark.parametrize(
    "profile", [_GPT_OSS_120B_PROFILE, _DSV4_PROFILE, _KIMI_K3_PROFILE]
)
def test_profile_without_raised_capacity_refuses_above_1024_at_ep8(profile) -> None:
    # The server admits EP8 chunks up to 2624 rows for DSv4.1 Flash; a model
    # whose profile keeps the default refuses them when its workspace is built.
    assert gluon_petit_max_tokens_per_rank(8, 2048) == 2048
    with pytest.raises(
        ValueError, match=f"EP8 keeps the default capacity for {profile.name}"
    ):
        gluon_petit_max_tokens_per_rank(8, 1025, profile)
    with (
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.dist.is_initialized", return_value=True
        ),
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.dist.get_world_size", return_value=8
        ),
        pytest.raises(
            ValueError, match=f"keeps the default capacity for {profile.name}"
        ),
    ):
        _get_workspace(torch.device("cuda", 0), profile, 2048)


def test_muse_spark_capacity_at_ep8() -> None:
    # The server admits EP8 chunks up to DSv4.1 Flash's 2624 rows; Muse Spark's
    # workspace admits its 2304-row chunk and its 2368-row limit and refuses
    # the next 32-row block, which would take its heap past 32-bit offsets.
    assert gluon_petit_largest_max_tokens_per_rank(8) == 2624
    assert _profile_largest_max_tokens_per_rank(_MUSE_SPARK_PROFILE, 8) == 2368
    assert gluon_petit_max_tokens_per_rank(8, 2304, _MUSE_SPARK_PROFILE) == 2304
    assert gluon_petit_max_tokens_per_rank(8, 2368, _MUSE_SPARK_PROFILE) == 2368
    for rows in (2369, 2400, 2560):
        assert gluon_petit_max_tokens_per_rank(8, rows) == -(-rows // 32) * 32
        with pytest.raises(
            ValueError,
            match="at most 2368 tokens per rank at world size 8 for muse_spark; "
            rf"got {rows} \(a larger capacity would exceed",
        ):
            gluon_petit_max_tokens_per_rank(8, rows, _MUSE_SPARK_PROFILE)
    with (
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.dist.is_initialized", return_value=True
        ),
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.dist.get_world_size", return_value=8
        ),
        pytest.raises(ValueError, match="at most 2368 tokens per rank"),
    ):
        _get_workspace(torch.device("cuda", 0), _MUSE_SPARK_PROFILE, 2400)
    petit_kernel = _import_petit_kernel()
    reserved, _ = _megamoe_config(
        petit_kernel, _MUSE_SPARK_PROFILE, 8, 2368
    ).heap_bytes()
    assert reserved == 4_252_979_200
    # Startup compiles one more variant at 2304 rows than at 1024: the plain
    # Triton class above 1024 rows.
    assert _megamoe_config(
        petit_kernel, _MUSE_SPARK_PROFILE, 8, 1024
    ).variant_token_counts() == (0, 1, 2, 12, 16, 128, 129, 1024)
    assert _megamoe_config(
        petit_kernel, _MUSE_SPARK_PROFILE, 8, 2304
    ).variant_token_counts() == (0, 1, 2, 12, 16, 128, 129, 1024, 1025)


@pytest.mark.parametrize(
    "world_size,rows,capacity",
    [
        (4, 1025, 1056),
        (4, 2048, 2048),
        (4, 4000, 4000),
        (4, 4096, 4096),
        (4, 7808, 7808),
        (4, 7809, None),
        (8, 1025, 1056),
        (8, 2048, 2048),
        (8, 2560, 2560),
        (8, 2624, 2624),
        (8, 2625, None),
        (8, 4096, None),
    ],
)
def test_capacity_rounds_to_sorted_blocks_and_refuses_above_heap_limit(
    world_size, rows, capacity
) -> None:
    if capacity is None:
        largest = 7808 if world_size == 4 else 2624
        with pytest.raises(ValueError, match=f"at most {largest} tokens per rank"):
            gluon_petit_max_tokens_per_rank(world_size, rows)
    else:
        assert gluon_petit_max_tokens_per_rank(world_size, rows) == capacity


@pytest.mark.parametrize(
    "profile,rows,expected",
    [
        # Stage-1 thresholds were tuned to 1024 rows; above them every count
        # keeps the 8-warp 64-row tile and the 56-CTA producer.
        (_DSV41_FLASH_PROFILE, 1024, ("cta56", "m64_w8")),
        (_DSV41_FLASH_PROFILE, 2048, ("cta56", "m64_w8")),
        (_DSV41_FLASH_PROFILE, 4096, ("cta56", "m64_w8")),
        (_DSV41_FLASH_DSPARK_PROFILE, 1024, ("cta56", "m64_w8")),
        (_DSV41_FLASH_DSPARK_PROFILE, 2048, ("cta56", "m64_w8")),
        (_DSV41_FLASH_DSPARK_PROFILE, 4096, ("cta56", "m64_w8")),
    ],
)
def test_ep4_launch_choice_above_1024_rows(profile, rows, expected) -> None:
    petit_kernel = _import_petit_kernel()
    config = _megamoe_config(petit_kernel, profile, 4, 4096)
    assert config.launch_choice(rows) == expected
    # The same count at the default capacity launches the same tile.
    if rows <= 1024:
        assert _megamoe_config(petit_kernel, profile, 4, 1024).launch_choice(
            rows
        ) == config.launch_choice(rows)


@pytest.mark.parametrize("profile", [_DSV41_FLASH_PROFILE, _DSV41_FLASH_DSPARK_PROFILE])
@pytest.mark.parametrize("rows", [1024, 1360, 2048, 2560])
def test_ep8_launch_choice_above_1024_rows(profile, rows) -> None:
    # At EP8 both profiles launch the 8-warp 64-row tile and the 56-CTA
    # producer from 1024 rows up to the raised 2560-row capacity.
    petit_kernel = _import_petit_kernel()
    config = _megamoe_config(petit_kernel, profile, 8, 2560)
    assert config.launch_choice(rows) == ("cta56", "m64_w8")


def test_raised_capacity_is_carried_into_state_and_apply() -> None:
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
    layer = SimpleNamespace(
        _moe_backend_state=_GluonPetitState(
            profile=profile,
            w13_weight=torch.empty(0),
            w2_weight=torch.empty(0),
            w13_scale=torch.empty(0),
            w2_scale=torch.empty(0),
            w13_bias=None,
            w2_bias=None,
            max_tokens_per_rank=2048,
        )
    )
    x = torch.empty((2049, profile.model_dim), dtype=torch.bfloat16)
    for kwargs, error in (
        ({"max_num_tokens_per_gpu": 2049}, "its 2048-token capacity"),
        ({"max_num_tokens_per_gpu": None}, "more than 2048 tokens on one rank"),
    ):
        with pytest.raises(ValueError, match=error):
            gluon_petit_mxfp4_megamoe_apply(
                plan={},
                x=x,
                w=layer,
                router_logits=torch.empty((0, 0), dtype=torch.bfloat16),
                topk_weights=torch.empty((2049, 1), dtype=torch.float32),
                topk_ids=torch.empty((2049, 1), dtype=torch.int32),
                num_tokens_global=2049,
                do_finalize=True,
                enable_pdl=False,
                low_latency=None,
                overlap_fn=None,
                **kwargs,
            )


def test_workspace_logs_profile_capacity_and_heap(caplog) -> None:
    petit_kernel = _import_petit_kernel()
    heap = object()
    inputs = object()
    caplog.set_level(logging.INFO, logger="tokenspeed_kernel.ops.moe.gluon")
    with (
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.dist.is_initialized", return_value=True
        ),
        mock.patch(
            "tokenspeed_kernel.ops.moe.gluon.dist.get_world_size", return_value=8
        ),
        mock.patch.object(
            petit_kernel, "create_vmm_symmetric_heap", return_value=heap
        ) as create_heap,
        mock.patch.object(
            petit_kernel.MegaMoeConfig, "input_views", return_value=inputs
        ),
    ):
        try:
            for _ in range(2):
                workspace = _get_workspace(
                    torch.device("cuda", 0), _DSV41_FLASH_PROFILE, 2560
                )
        finally:
            _workspace_cache.clear()
    create_heap.assert_called_once_with(8)
    assert workspace.heap is heap and workspace.inputs is inputs
    reserved, physical = workspace.config.heap_bytes()
    assert reserved < 1 << 32
    # One line per workspace, not per layer.
    assert [r.getMessage() for r in caplog.records] == [
        "Gluon Petit workspace: profile=dsv41_flash world_size=8 "
        f"max_tokens_per_rank=2560 heap_reserved_bytes={reserved} "
        f"heap_physical_bytes={physical}"
    ]


def test_startup_compile_logs_profile_capacity_and_counts(caplog) -> None:
    config = mock.Mock(world_size=8)
    config.compile_variants.return_value = (0, 1, 2, 256, 257, 512, 513)
    workspace = SimpleNamespace(config=config, heap=object(), inputs=object())
    state = _GluonPetitState(
        profile=_DSV41_FLASH_PROFILE,
        w13_weight=torch.empty(0),
        w2_weight=torch.empty(0),
        w13_scale=torch.empty(0),
        w2_scale=torch.empty(0),
        w13_bias=None,
        w2_bias=None,
        max_tokens_per_rank=2560,
    )
    caplog.set_level(logging.INFO, logger="tokenspeed_kernel.ops.moe.gluon")
    _compiled_variant_keys.clear()
    try:
        for _ in range(2):
            _compile_variants(torch.device("cuda", 0), state, workspace)
    finally:
        _compiled_variant_keys.clear()
    config.compile_variants.assert_called_once()
    (message,) = [r.getMessage() for r in caplog.records]
    assert re.fullmatch(
        r"Gluon Petit startup compile: profile=dsv41_flash world_size=8 "
        r"max_tokens_per_rank=2560 variants=7 counts=\[0,1,2,256,257,512,513\] "
        r"seconds=\d+\.\d",
        message,
    ), message


@pytest.mark.parametrize("profile", [_DSV41_FLASH_PROFILE, _DSV41_FLASH_DSPARK_PROFILE])
def test_stage1_n128_switch_selects_tile_and_variants(profile) -> None:
    """C5 (TOKENSPEED_PETIT_STAGE1_N128): counts up to the threshold take the
    32x128 stage-1 tile, a separate compile variant; stage 2 keeps its tile.
    Off (-1, the default) every count keeps the old launch choice."""
    import sys

    petit_kernel = _import_petit_kernel()
    import lib.moe.rocm.mega_moe as forwarding

    module = sys.modules[forwarding.MegaMoESolutionAdapter.__module__]
    config = _megamoe_config(petit_kernel, profile, 4, 1024)
    with mock.patch.object(module, "kStage1N128MaxTokens", -1):
        off_counts = config.variant_token_counts()
        off_choice = {n: config.launch_choice(n) for n in (0, 1, 6, 8, 9, 24)}
    assert all(tile == "m32" for _, tile in off_choice.values())
    with mock.patch.object(module, "kStage1N128MaxTokens", 8):
        for n in (0, 1, 6, 8, 9, 24):
            geometry, tile = config.launch_choice(n)
            assert geometry == off_choice[n][0]
            assert tile == ("m32_n128" if n <= 8 else "m32")
        # The first count above the threshold in each Triton integer class
        # (plain 9, multiple-of-16 16) leaves the N=128 tile.
        assert config.variant_token_counts() == tuple(sorted({*off_counts, 9, 16}))
        adapter = module._MegaMoESolution(config._solution_id_for_tokens(1))
        s1, s2, combine = adapter.Kernels(True, False, False, True, n128=True)
        b1, b2, b_combine = adapter.Kernels(True, False, False, True)
        assert s1.kStage1TileCount == 2 * b1.kStage1TileCount
        assert s1.cache_key != b1.cache_key
        assert s2.cache_key == b2.cache_key and combine.cache_key == b_combine.cache_key
    with mock.patch.object(module, "kStage1N128MaxTokens", 4096):
        # The 64-row tiles keep their counts whatever the threshold.
        assert (
            config.launch_choice(480)[1] == "m64_w4"
            or profile is not _DSV41_FLASH_PROFILE
        )
        assert config.launch_choice(1009)[1] == "m64_w8"


@pytest.mark.parametrize("profile", [_DSV41_FLASH_PROFILE, _DSV41_FLASH_DSPARK_PROFILE])
def test_stage2_n128_config_replaces_stage2_only(profile) -> None:
    """C5b (TOKENSPEED_PETIT_STAGE2_N128): MegaMoEStage2N128Config changes the
    stage-2 kernel alone. Every site that follows stage 2's column count
    agrees on 128 (scheduler tiles, epilogue tile and LDS words, accumulator
    fragments, W2 tile), the stage-1 arrival mask and the stage-1 and
    combine kernels stay the base's, and the epilogue's LDS column map and
    write-back cover every tile element exactly once (both widths)."""
    import sys

    petit_kernel = _import_petit_kernel()
    import lib.moe.rocm.mega_moe as forwarding

    module = sys.modules[forwarding.MegaMoESolutionAdapter.__module__]
    config = _megamoe_config(petit_kernel, profile, 4, 1024)
    adapter = module._MegaMoESolution(config._solution_id_for_tokens(1))
    b1, b2, b_combine = adapter.Kernels(True, False, False, True)
    s1, s2, combine = adapter.Kernels(True, False, False, True, stage2_n128=True)
    n1, n2, n_combine = adapter.Kernels(
        True, False, False, True, n128=True, stage2_n128=True
    )
    assert s1.cache_key == b1.cache_key and combine.cache_key == b_combine.cache_key
    assert n_combine.cache_key == b_combine.cache_key and n2.cache_key == s2.cache_key
    assert s2.cache_key != b2.cache_key
    assert combine.Config is adapter.Config and s2.Config is not adapter.Config
    assert (b2.Config.kGroupN, s2.Config.kGroupN) == (256, 128)
    assert s2.Scheduler.kLinear2Tiles == 2 * b2.Scheduler.kLinear2Tiles
    assert s2.Scheduler.kLinear1Tiles == b2.Scheduler.kLinear1Tiles
    assert s2.kStage1TileCount == b2.kStage1TileCount  # L2 arrival mask
    assert s2.kStage2GridBlocks == b2.kStage2GridBlocks
    epi, base_epi = s2.Stage2Epilogue, b2.Stage2Epilogue
    assert (epi.kTileCols, epi.kAccumFragments, epi.kWaveCols) == (128, 4, 32)
    assert (base_epi.kTileCols, base_epi.kAccumFragments, base_epi.kWaveCols) == (
        256,
        8,
        64,
    )
    assert epi.kShmWords == 32 * 128 // 2 + 2 * 32
    assert s2.kComputeWords >= epi.kShmWords and s2.kShmWords >= s2.kComputeWords
    assert s2.W2Weights.W2.kGroupN == 128 and b2.W2Weights.W2.kGroupN == 256
    assert s2.W13Weights is b2.W13Weights
    # The switch: off names no stage2_n128 (the old Kernels entries), on
    # names it for bias-free solutions only.
    with mock.patch.object(module, "kStage2N128", False):
        assert not adapter.Stage2N128()
    with mock.patch.object(module, "kStage2N128", True):
        assert adapter.Stage2N128()
    for e in (epi, base_epi):
        # WriteShm: (wid, lane, mi, ni, component) -> LDS (row, col), BF16 units.
        written = []
        for wid in range(4):
            for wtid in range(64):
                q, r = wtid // 16, wtid % 16
                for mi in range(2):
                    for ni in range(e.kAccumFragments // 2):
                        for component in range(4):
                            row = mi * 16 + r
                            col = wid * e.kWaveCols + ni * 16 + q * 4 + component
                            written.append((row, col))
        assert sorted(written) == [
            (row, col) for row in range(32) for col in range(e.kTileCols)
        ]
        # Route write-back (route_output.py): per row, kTileCols // 128 halves of
        # 64 lanes x 2 BF16 columns.
        cols = sorted(
            (half * 64 + wtid) * 2 + k
            for half in range(e.kTileCols // 128)
            for wtid in range(64)
            for k in range(2)
        )
        assert cols == list(range(e.kTileCols))
        # Single-rank write-back (op_stages.py): kTileCols // 64 x 32 lanes x 2.
        cols = sorted(
            nr * 64 + n_lane * 2 + k
            for nr in range(e.kTileCols // 64)
            for n_lane in range(32)
            for k in range(2)
        )
        assert cols == list(range(e.kTileCols))


def test_stage1_n128_received_experts_rule() -> None:
    """C5 selector: N=128 pays when it saves half a tile round (pure rule)."""
    import lib.moe.rocm.mega_moe as forwarding

    _import_petit_kernel()
    pays = forwarding.Stage1N128PaysForExperts
    # DSv4.1 Flash at EP4: 2560 / 256 = 10 N=256 tiles per expert, 256 CTAs.
    chosen = [e for e in range(0, 97) if pays(e, 10, 256)]
    assert chosen == [
        *range(1, 13),
        *range(26, 39),
        *range(52, 65),
        *range(77, 90),
    ]
    estimate = forwarding.EstimatedReceivedExperts
    # rows, top_k, experts, ranks -> expected distinct local experts
    assert estimate(0, 6, 384, 4) == 0
    assert estimate(6, 6, 384, 4) == 9  # DP4 / tp4ep4 batch 1 (1 + 5 draft)
    assert estimate(12, 6, 384, 4) == 17  # batch 2
    assert estimate(24, 6, 384, 4) == 30  # batch 4
    assert estimate(10_000, 6, 384, 4) == 96  # capped at the local experts
    assert estimate(24, 3, 128, 4) == 14  # DSpark draft, 6 rows on 4 ranks
    # Few rows: rows * top_k / ranks (no collisions).
    assert estimate(2, 6, 384, 4) == 3


@pytest.mark.parametrize(
    "profile,expected",
    [
        (
            _DSV41_FLASH_PROFILE,
            {
                0: False,
                4: True,
                6: True,
                8: True,
                12: False,
                18: False,
                24: True,
                48: False,
                96: False,
            },
        ),
        (
            _DSV41_FLASH_DSPARK_PROFILE,
            {0: False, 4: True, 6: True, 12: True, 24: False, 48: False},
        ),
    ],
)
def test_stage1_n128_selector_by_received_experts(profile, expected) -> None:
    """C5 with TOKENSPEED_PETIT_STAGE1_N128_SELECT=experts: one call-wide tile
    from the collective's row count; the row rule is off, a rank still takes
    N=128 only where its own count admits the 32-row tile; the warmup compiles
    the N=128 tile for every 32-row-tile count."""
    import sys

    petit_kernel = _import_petit_kernel()
    import lib.moe.rocm.mega_moe as forwarding

    module = sys.modules[forwarding.MegaMoESolutionAdapter.__module__]
    config = _megamoe_config(petit_kernel, profile, 4, 1024)
    with mock.patch.object(module, "kStage1N128ByExperts", False):
        assert config.stage1_n128_for_global_rows(6) is None
    with (
        mock.patch.object(module, "kStage1N128ByExperts", True),
        mock.patch.object(module, "kStage1N128MaxTokens", -1),
    ):
        got = {rows: config.stage1_n128_for_global_rows(rows) for rows in expected}
        assert got == expected
        adapter = module._MegaMoESolution(config._solution_id_for_tokens(1))
        min_m64 = adapter.Config.kM64MinTokens
        assert adapter.SelectStage1N128(0, True)  # a 0-row rank follows the call
        assert not adapter.SelectStage1N128(min_m64, True)  # 64-row tile kept
        assert not adapter.SelectStage1N128(1, False)
        assert not adapter.SelectStage1N128(1, None)  # no row rule
        # Variant keys keep the switch-off set; compile_variants adds the
        # N=128 tile for each 32-row-tile count.
        off_counts = config.variant_token_counts()
        launches = []
        with mock.patch.object(
            type(config),
            "_launch",
            side_effect=lambda *a, **k: launches.append(
                (a[5], k["reduce_routes"], k["stage1_n128"])
            ),
        ):
            views = SimpleNamespace(
                tokens=torch.empty((1024, 8)),
                scales=torch.empty((1024, 1)),
                expert_ids=torch.empty((1024, 1)),
                expert_weights=torch.empty((1024, 1)),
            )
            config.compile_variants(
                None,
                None,
                None,
                None,
                None,
                w13_bias=None,
                w2_bias=None,
                inputs=views,
            )
        m32 = [n for n in off_counts if config.launch_choice(n)[1] == "m32"]
        assert m32 and len(m32) < len(off_counts)
        assert sorted({n for n, _, s in launches if s}) == m32
        assert sorted({n for n, _, s in launches if s is None}) == list(off_counts)
        assert len(launches) == 2 * (len(off_counts) + len(m32))
    with (
        mock.patch.object(module, "kStage1N128ByExperts", False),
        mock.patch.object(module, "kStage1N128MaxTokens", -1),
    ):
        launches.clear()
        with mock.patch.object(
            type(config),
            "_launch",
            side_effect=lambda *a, **k: launches.append(k["stage1_n128"]),
        ):
            config.compile_variants(
                None,
                None,
                None,
                None,
                None,
                w13_bias=None,
                w2_bias=None,
                inputs=views,
            )
        assert launches == [None] * (2 * len(off_counts))


def _synthetic_received_experts(profile_name, counts):
    """Distinct local experts each rank receives under the Petit tests'
    synthetic routing (``_routing`` of test_gluon_petit_distributed.py, the
    ``--routing test`` of petit_layer.py), for per-rank row ``counts``."""
    import importlib.util
    import pathlib
    import sys

    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "amd/gfx950/ops/moe/test_gluon_petit_distributed.py"
    )
    spec = importlib.util.spec_from_file_location("_petit_distributed_routing", path)
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    profile = module._PROFILES[profile_name]
    ws, local = len(counts), profile.experts // len(counts)
    received = [set() for _ in counts]
    for rank, count in enumerate(counts):
        ids, _ = module._routing(count, rank, profile, ws, torch.device("cpu"))
        for expert in ids.flatten().tolist():
            received[expert // local].add(expert)
    return [len(r) for r in received]


def _selector_cases():
    """Every decode count r0002-f3 measured, and rows 1-64 in both layouts:
    DP4 (all rows on one rank) and tp4ep4 (replicated rows split in
    ceil(R/4) chunks), for the target and the DSpark draft."""
    cases = []
    for profile in ("dsv41_flash", "dsv41_flash_dspark"):
        for rows in range(1, 65):
            chunk = -(-rows // 4)
            cases.append((profile, "dp4", [rows, 0, 0, 0]))
            cases.append(
                (
                    profile,
                    "tp4ep4",
                    [max(0, min(chunk, rows - i * chunk)) for i in range(4)],
                )
            )
    return cases


def selector_table():
    """Rows of (profile, layout, counts, estimate, collision-free, delivered
    per rank, old decision, new decision, right decision at the delivered
    counts). Print with ``python3 -c`` from the test directory."""
    import sys

    _import_petit_kernel()
    import lib.moe.rocm.mega_moe as forwarding

    module = sys.modules[forwarding.MegaMoESolutionAdapter.__module__]
    shapes = {"dsv41_flash": (384, 6), "dsv41_flash_dspark": (128, 3)}
    tiles, ctas = 2560 // 256, 256
    rows = []
    for profile, layout, counts in _selector_cases():
        experts, top_k = shapes[profile]
        total = sum(counts)
        estimate = module.EstimatedReceivedExperts(total, top_k, experts, 4)
        low, high = module.ReceivedExpertsBounds(total, top_k, experts, 4)
        delivered = _synthetic_received_experts(profile, counts)
        old = module.Stage1N128PaysForExperts(estimate, tiles, ctas)
        new = high > 0 and all(
            module.Stage1N128PaysForExperts(e, tiles, ctas)
            for e in range(low, high + 1)
        )
        right = {
            module.Stage1N128PaysForExperts(e, tiles, ctas) for e in delivered if e
        }
        rows.append(
            (profile, layout, counts, estimate, high, delivered, old, new, right)
        )
    return rows


def test_stage1_n128_selector_right_at_synthetic_counts() -> None:
    """The fixed selector (``ReceivedExpertsBounds``: N=128 only where the
    tile-round rule takes it from the collision estimate to the
    collision-free count) never takes N=128 where the rule at the delivered
    per-rank count says N=256, at rows 1-64 in DP4 and tp4ep4 for the target
    and the draft; it keeps the measured wins (DP4/tp4ep4 b1, b4) and fixes
    the measured loss (R=64: estimate 61, delivered 96, r0002-f3)."""
    table = selector_table()
    for profile, layout, counts, _, high, delivered, _, new, right in table:
        if new:
            assert right == {True}, (profile, layout, counts, delivered)
        assert max(delivered) <= high + 1, (profile, layout, counts, delivered)
    by = {(p, l, sum(c)): (old, new) for p, l, c, _, _, _, old, new, _ in table}
    assert by[("dsv41_flash", "dp4", 64)] == (True, False)  # the f3 miss
    assert by[("dsv41_flash", "tp4ep4", 64)] == (True, False)
    assert by[("dsv41_flash", "tp4ep4", 32)] == (True, False)  # measured tie
    for layout in ("dp4", "tp4ep4"):
        assert by[("dsv41_flash", layout, 6)] == (True, True)  # b1 win kept
        assert by[("dsv41_flash", layout, 24)] == (True, True)  # b4 win kept
        assert by[("dsv41_flash", layout, 12)] == (False, False)  # b2
    assert by[("dsv41_flash_dspark", "tp4ep4", 24)] == (False, False)  # draft 6x4


@pytest.mark.parametrize("selected", (None, True, False))
@pytest.mark.parametrize("do_finalize", (True, False))
def test_apply_passes_call_wide_stage1_tile(selected, do_finalize) -> None:
    """The apply asks the selector with num_tokens_global and passes its
    answer to the launch unchanged (None: selector off, row rule)."""
    from tokenspeed_kernel.ops.moe import gluon as petit

    profile = _Profile(
        name="test",
        num_experts=8,
        top_k=2,
        model_dim=64,
        logical_intermediate=32,
        inter_dim=32,
        has_bias=False,
        activation="silu",
    )
    heap = SimpleNamespace(
        tokens=torch.empty((4, 32), dtype=torch.uint8),
        scales=torch.empty((4, 2), dtype=torch.uint8),
        expert_ids=torch.empty((4, 2), dtype=torch.int32),
        expert_weights=torch.empty((4, 2), dtype=torch.float32),
    )
    config = mock.Mock()
    config.run.side_effect = lambda *args, **kwargs: kwargs["out"]
    config.run_routes.side_effect = lambda *args, **kwargs: torch.empty(0)
    config.stage1_n128_for_global_rows.return_value = selected
    workspace = SimpleNamespace(config=config, heap=object(), inputs=heap)
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
    x = torch.zeros((1, profile.model_dim), dtype=torch.bfloat16)
    with mock.patch(
        "tokenspeed_kernel.ops.moe.gluon._get_workspace", return_value=workspace
    ):
        petit.gluon_petit_mxfp4_megamoe_apply(
            plan={},
            x=x,
            w=layer,
            router_logits=torch.empty((1, 0), dtype=torch.bfloat16),
            topk_weights=torch.ones((1, 2), dtype=torch.float32),
            topk_ids=torch.zeros((1, 2), dtype=torch.int32),
            num_tokens_global=7,
            max_num_tokens_per_gpu=4,
            do_finalize=do_finalize,
            enable_pdl=False,
            low_latency=None,
            overlap_fn=None,
        )
    config.stage1_n128_for_global_rows.assert_called_once_with(7)
    launch = config.run if do_finalize else config.run_routes
    assert launch.call_args.kwargs["stage1_n128"] is selected


def _fuse_post_bound(old, new, routed):
    """C3's rounding bound. Old: bf16(fp32(bf16(A)) + s), A the FP32 route
    sum; new: bf16(fl(((s + r0) + ...) + r5)). With bf16 half-ulp u = 2^-9:
    |bf16(A) - A| <= u|A|, each final rounding <= u|A + s|, and the two FP32
    orders differ by <= 6 * 2^-24 * (|s| + sum |r|), far below u. So
    |old - new| <= u|A| + 2u|A + s| (+ FP32 noise); the check doubles it:
    2^-8 |A| + 2^-7 max(|old|, |new|) + 2^-120."""
    old, new, routed = old.float(), new.float(), routed.float()
    return (old - new).abs() <= (
        routed.abs() * 2.0**-8
        + torch.maximum(old.abs(), new.abs()) * 2.0**-7
        + 2.0**-120
    )


def test_fuse_post_rounding_bound_on_cpu() -> None:
    """C3 changes the rounding of the DP4 post add; emulate both orders in
    PyTorch (FP32 adds round to nearest even like v_pk_add_f32) and check
    the stated bound, that the change is not bit-exact, and that a zero
    residual is bit-exact (the residual load adds an exact 0)."""
    gen = torch.Generator().manual_seed(0)
    routes = (torch.randn((64, 6, 512), generator=gen) * 0.05).to(torch.bfloat16)
    shared = (torch.randn((64, 512), generator=gen) * 0.5).to(torch.bfloat16)

    def fused(s):
        acc = s.float()
        for k in range(routes.shape[1]):
            acc = acc + routes[:, k].float()
        return acc.to(torch.bfloat16)

    acc = torch.zeros((64, 512))
    for k in range(routes.shape[1]):
        acc = acc + routes[:, k].float()
    routed = acc.to(torch.bfloat16)
    old = torch.add(routed, shared)
    new = fused(shared)
    assert bool(_fuse_post_bound(old, new, acc).all())
    assert not torch.equal(old, new)  # not bit-exact: needs the GSM8K gate
    assert torch.equal(fused(torch.zeros_like(shared)), routed)


def test_fuse_post_combine_variant_and_apply_plumbing() -> None:
    """C3: the residual combine is its own kernel (the off path keeps the old
    cache keys); inside petit_combine_post the apply passes residual, ready
    and out to run(); outside it passes neither."""
    import sys

    petit_kernel = _import_petit_kernel()
    import lib.moe.rocm.mega_moe as forwarding
    from tokenspeed_kernel.ops.moe import gluon as petit

    module = sys.modules[forwarding.MegaMoESolutionAdapter.__module__]
    config = _megamoe_config(petit_kernel, _DSV41_FLASH_PROFILE, 4, 1024)
    adapter = module._MegaMoESolution(config._solution_id_for_tokens(1))
    b1, b2, b_combine = adapter.Kernels(True, False, False, True)
    r1, r2, r_combine = adapter.Kernels(True, False, False, True, add_residual=True)
    assert r1.cache_key == b1.cache_key and r2.cache_key == b2.cache_key
    assert r_combine.kAddResidual and not b_combine.kAddResidual
    assert r_combine.cache_key != b_combine.cache_key
    assert r_combine.Reducer.kAddResidual and r_combine.Reducer.kReduceRoutes
    from lib.moe.rocm.ops.mega_moe.route_output import SourceRouteReducer

    with pytest.raises(AssertionError):  # a residual needs reduced routes
        SourceRouteReducer(adapter.Config, 128, 512, False, True)

    profile = _Profile(
        name="test",
        num_experts=8,
        top_k=2,
        model_dim=64,
        logical_intermediate=32,
        inter_dim=32,
        has_bias=False,
        activation="silu",
    )
    heap = SimpleNamespace(
        tokens=torch.empty((4, 32), dtype=torch.uint8),
        scales=torch.empty((4, 2), dtype=torch.uint8),
        expert_ids=torch.empty((4, 2), dtype=torch.int32),
        expert_weights=torch.empty((4, 2), dtype=torch.float32),
    )
    run_config = mock.Mock()
    run_config.run.side_effect = lambda *args, **kwargs: kwargs["out"]
    run_config.stage1_n128_for_global_rows.return_value = None
    workspace = SimpleNamespace(config=run_config, heap=object(), inputs=heap)
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
    x = torch.zeros((1, profile.model_dim), dtype=torch.bfloat16)

    def apply():
        return petit.gluon_petit_mxfp4_megamoe_apply(
            plan={},
            x=x,
            w=layer,
            router_logits=torch.empty((1, 0), dtype=torch.bfloat16),
            topk_weights=torch.ones((1, 2), dtype=torch.float32),
            topk_ids=torch.zeros((1, 2), dtype=torch.int32),
            num_tokens_global=1,
            max_num_tokens_per_gpu=1,
            do_finalize=True,
            enable_pdl=False,
            low_latency=None,
            overlap_fn=None,
        )

    with mock.patch(
        "tokenspeed_kernel.ops.moe.gluon._get_workspace", return_value=workspace
    ):
        apply()
        kwargs = run_config.run.call_args.kwargs
        assert "residual" not in kwargs and "residual_ready" not in kwargs
        residual = torch.ones((1, profile.model_dim), dtype=torch.bfloat16)
        target = torch.empty_like(residual)
        ready = object()
        with petit.petit_combine_post(residual, out=target, ready=ready):
            got = apply()
        kwargs = run_config.run.call_args.kwargs
        assert kwargs["residual"] is residual and kwargs["residual_ready"] is ready
        assert got is target and kwargs["out"] is target
        assert not petit._PETIT_COMBINE_POST  # the scope is popped


def test_stage1_n128_received_experts_rule_ep8() -> None:
    """C5 selector at EP8 (48 local target experts, 16 draft): DP8 decode
    with 1 + 5 draft rows per request routes R = 6 x concurrency rows."""
    import lib.moe.rocm.mega_moe as forwarding

    _import_petit_kernel()
    pays = forwarding.Stage1N128PaysForExperts
    assert [e for e in range(0, 49) if pays(e, 10, 256)] == [
        *range(1, 13),
        *range(26, 39),
    ]
    bounds = forwarding.ReceivedExpertsBounds
    assert {r: bounds(r, 6, 384, 8) for r in (6, 12, 24, 48, 96)} == {
        6: (4, 5),
        12: (8, 9),
        24: (15, 18),
        48: (25, 36),
        96: (37, 48),
    }
    assert {r: bounds(r, 3, 128, 8) for r in (1, 2, 4, 8, 16)} == {
        1: (1, 1),
        2: (1, 1),
        4: (1, 2),
        8: (3, 3),
        16: (5, 6),
    }
