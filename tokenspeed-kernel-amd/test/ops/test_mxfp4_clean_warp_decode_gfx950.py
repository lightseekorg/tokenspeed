# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "AMD GFX950 is required for MXFP4 warp-decode tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950  # noqa: E402
from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (  # noqa: E402
    gluon_mxfp_dynamic_mxfp4_fused_moe,
    gluon_mxfp_precomputed_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950 import (  # noqa: E402
    gluon_mxfp4_moe_decode,
    invoke_sigmoid_bias_topk_route_gluon,
    invoke_softmax_topk_route_gluon,
)
from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)


def _make_weights(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(910603)
    nibble_values = torch.tensor((0, 1, 2, 9, 10), device=device, dtype=torch.uint8)

    def packed(shape):
        lo = nibble_values[
            torch.randint(
                0, len(nibble_values), shape, device=device, generator=generator
            )
        ]
        hi = nibble_values[
            torch.randint(
                0, len(nibble_values), shape, device=device, generator=generator
            )
        ]
        return lo | (hi << 4)

    w13 = packed((num_experts, 2 * intermediate_size, hidden_size // 2))
    w2 = packed((num_experts, hidden_size, intermediate_size // 2))
    w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        127,
        device=device,
        dtype=torch.uint8,
    )
    w2_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 32),
        127,
        device=device,
        dtype=torch.uint8,
    )
    return w13, w13_scale, w2, w2_scale


def test_dynamic_mxfp4_precomputed_tiny_m_prefers_direct_over_mfma(
    monkeypatch: pytest.MonkeyPatch,
):
    # Below _PRECOMPUTED_MFMA_MIN_M the dispatch must try the direct decode
    # kernel and must not fall through to the precomputed-MFMA decode.
    hidden = torch.empty((1, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((1, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    topk_ids = torch.empty((1, 1), device="cuda", dtype=torch.int32)
    topk_weights = torch.empty((1, 1), device="cuda", dtype=torch.float32)
    direct_sentinel = torch.empty_like(hidden)
    direct_calls = 0
    mfma_calls = 0

    def fake_direct_decode(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        assert kwargs["precomputed_topk_ids"] is topk_ids
        assert kwargs["precomputed_topk_weights"] is topk_weights
        return direct_sentinel

    def fake_mfma_decode(*args, **kwargs):
        nonlocal mfma_calls
        mfma_calls += 1
        return torch.empty_like(hidden)

    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_direct_mfma_decode",
        fake_direct_decode,
    )
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_mfma_decode",
        fake_mfma_decode,
    )
    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_ids=topk_ids,
        precomputed_topk_weights=topk_weights,
    )

    assert out is direct_sentinel
    assert direct_calls == 1
    assert mfma_calls == 0


def test_dynamic_mxfp4_precomputed_default_uses_mfma_for_medium_m(
    monkeypatch: pytest.MonkeyPatch,
):
    hidden = torch.empty((4, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((4, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    topk_ids = torch.empty((4, 1), device="cuda", dtype=torch.int32)
    topk_weights = torch.empty((4, 1), device="cuda", dtype=torch.float32)
    sentinel = torch.empty_like(hidden)
    mfma_calls = 0

    def fake_mfma_decode(*args, **kwargs):
        nonlocal mfma_calls
        mfma_calls += 1
        assert kwargs["precomputed_topk_ids"] is topk_ids
        assert kwargs["precomputed_topk_weights"] is topk_weights
        return sentinel

    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_mfma_decode",
        fake_mfma_decode,
    )
    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_ids=topk_ids,
        precomputed_topk_weights=topk_weights,
    )

    assert out is sentinel
    assert mfma_calls == 1


def test_dynamic_mxfp4_package_prefill_runs_on_caller_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    # The package-prefill bridge uses the in-house ``gluon_moe_sorting``, which
    # runs on the caller's stream, so the bridge must execute entirely on that
    # stream with no cross-stream fence or ``record_stream`` fixup.
    hidden = torch.empty((128, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((128, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    caller_stream = torch.cuda.Stream()

    class Sentinel:
        def record_stream(self, stream):  # pragma: no cover - must not be called
            raise AssertionError(
                "package prefill must not need cross-stream record_stream"
            )

    sentinel = Sentinel()
    observed = {}

    def fake_package_prefill(*args, **kwargs):
        observed["stream"] = torch.cuda.current_stream()
        return sentinel

    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_gluon_package_mxfp4_prefill",
        fake_package_prefill,
    )

    with torch.cuda.stream(caller_stream):
        out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
            hidden,
            router,
            dummy_w,
            dummy_w,
            w13_mx_scale=dummy_scale,
            w2_mx_scale=dummy_scale,
            top_k=1,
            correction_bias=None,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
        )

    assert out is sentinel
    # The bridge ran on the caller's stream, not the default stream.
    assert observed["stream"] == caller_stream
    assert observed["stream"] != torch.cuda.default_stream()


def test_package_exposes_prefill_stage_entry_points():
    from tokenspeed_kernel_amd.ops.moe import gluon_a4w4_gfx950 as package
    from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950 import (
        invoke_gluon_mxfp4_moe_stage1,
        invoke_gluon_mxfp4_moe_stage2_1x2,
    )

    assert not hasattr(fused_mxfp_gfx950, "_maybe_runtime_mxfp4_warp_decode")
    assert not hasattr(package, "invoke_stage1_warp_decode_gluon")
    assert not hasattr(package, "invoke_stage2_warp_decode_gluon")
    assert invoke_gluon_mxfp4_moe_stage1.__module__.endswith(".prefill_stage1")
    assert invoke_gluon_mxfp4_moe_stage2_1x2.__module__.endswith(".prefill_stage2")


def _make_preprocessed_layer(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: str,
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.quant_config = type("QuantConfig", (), {})()
    layer.quant_config.use_dynamic_mxfp4_activations = True
    layer.w13_input_layout = "concatenated"
    layer.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale.clone(), requires_grad=False)
    layer.w13_weight_bias = torch.nn.Parameter(
        torch.zeros(num_experts, 2 * intermediate_size, device=device),
        requires_grad=False,
    )
    layer.w2_weight_bias = torch.nn.Parameter(
        torch.zeros(num_experts, hidden_size, device=device),
        requires_grad=False,
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, layer)
    return layer


def _precomputed_mfma_expected(
    hidden: torch.Tensor,
    router: torch.Tensor,
    layer: torch.nn.Module,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    # Force the MFMA decode kernel below its tuned batch-size gate so it can be
    # used as the bit-exact reference for the direct/route-owned decode tests.
    out = fused_mxfp_gfx950._maybe_precomputed_mxfp4_mfma_decode(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        w13_bias=None,
        w2_bias=None,
        out_dtype=torch.bfloat16,
        max_m=8,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
        min_m=1,
    )
    assert out is not None
    return out


def _softmax_topk_reference(
    router: torch.Tensor,
    topk: int,
    *,
    correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(router.float(), dim=-1)
    choice = scores
    if correction_bias is not None:
        choice = choice + correction_bias.float().unsqueeze(0)
    _, topk_ids = torch.topk(choice, k=topk, dim=-1, sorted=True)
    topk_weights = scores.gather(1, topk_ids)
    if normalize_topk_weights:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


def _sigmoid_bias_topk_reference(
    router: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(router.float()).to(router.dtype)
    _, topk_ids = torch.topk(
        scores.float() + correction_bias.float().unsqueeze(0),
        k=topk,
        dim=-1,
        sorted=True,
    )
    topk_weights = scores.gather(1, topk_ids)
    if normalize_topk_weights:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * routed_scaling_factor
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_stable_topk_uses_smaller_index_for_exact_ties(dtype: torch.dtype):
    values = torch.tensor([[1.0, 3.0, 3.0, -2.0, 3.0, 0.5]], device="cuda", dtype=dtype)

    selected, ids = fused_mxfp_gfx950._stable_topk_smaller_index(values, 4)

    torch.testing.assert_close(
        ids,
        torch.tensor([[1, 2, 4, 0]], device="cuda", dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        selected,
        torch.tensor([[3.0, 3.0, 3.0, 1.0]], device="cuda", dtype=dtype),
        rtol=0,
        atol=0,
    )


def test_stable_topk_preserves_non_tied_values_and_order():
    generator = torch.Generator(device="cuda").manual_seed(20260712)
    values = torch.randn(
        (32, 384), device="cuda", dtype=torch.float32, generator=generator
    )
    expected_values, expected_ids = torch.topk(values, 8, dim=-1, sorted=True)

    actual_values, actual_ids = fused_mxfp_gfx950._stable_topk_smaller_index(
        values, 8, dim=-1, sorted=True
    )

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stable_topk_is_cuda_graph_capturable(dtype: torch.dtype):
    """Regression: stable top-k must not copy CPU->CUDA (illegal under capture).

    The prior form built the tie-break masks with ``raw.new_tensor(<int>)``,
    which materializes a CPU tensor and copies it to the GPU -- this crashed
    CUDA-graph capture in the runtime model tests.
    """
    values = torch.randn((8, 128), device="cuda", dtype=dtype)
    eager_v, eager_i = fused_mxfp_gfx950._stable_topk_smaller_index(
        values, 8, dim=-1, sorted=True
    )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fused_mxfp_gfx950._stable_topk_smaller_index(values, 8, dim=-1, sorted=True)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cap_v, cap_i = fused_mxfp_gfx950._stable_topk_smaller_index(
            values, 8, dim=-1, sorted=True
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(cap_i, eager_i, rtol=0, atol=0)
    torch.testing.assert_close(cap_v, eager_v, rtol=0, atol=0)


def test_biased_grouped_route_is_repeatable_with_bf16_ties():
    num_tokens = 512
    num_experts = 384
    topk = 8
    generator = torch.Generator(device="cuda").manual_seed(603)
    logits = torch.randn(
        (num_tokens, num_experts),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    correction_bias = torch.zeros((num_experts,), device="cuda", dtype=torch.bfloat16)

    expected_weights, expected_ids = fused_mxfp_gfx950._biased_grouped_topk_reference(
        logits,
        correction_bias,
        topk,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    for _ in range(4):
        actual_weights, actual_ids = fused_mxfp_gfx950._biased_grouped_topk_reference(
            logits,
            correction_bias,
            topk,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
        )
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_softmax_topk_route_gluon_matches_reference():
    device = "cuda"
    router = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75, -1.0, 0.5, -0.25, 1.25],
            [-0.75, 0.5, 1.5, -0.25, 0.0, 1.0, -1.5, 0.25],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    correction_bias = torch.tensor(
        [0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3],
        device=device,
        dtype=torch.float32,
    )
    topk = 3
    expected_ids, expected_weights = _softmax_topk_reference(
        router,
        topk,
        correction_bias=correction_bias,
        routed_scaling_factor=1.75,
        normalize_topk_weights=True,
    )

    topk_ids, topk_weights = invoke_softmax_topk_route_gluon(
        router,
        topk,
        correction_bias=correction_bias,
        routed_scaling_factor=1.75,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights, atol=5e-3, rtol=5e-3)


def test_sigmoid_bias_topk_route_gluon_matches_reference():
    device = "cuda"
    router = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75, -1.0, 0.5, -0.25, 1.25],
            [-0.75, 0.5, 1.5, -0.25, 0.0, 1.0, -1.5, 0.25],
        ],
        device=device,
        dtype=torch.float32,
    )
    correction_bias = torch.tensor(
        [0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3],
        device=device,
        dtype=torch.float32,
    )
    topk = 3
    expected_ids, expected_weights = _sigmoid_bias_topk_reference(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )

    topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_gluon(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights, atol=5e-3, rtol=5e-3)


def test_sigmoid_bias_topk_route_gluon_matches_bf16_kimi_shape():
    device = "cuda"
    num_tokens = 8
    num_experts = 384
    topk = 8
    generator = torch.Generator(device=device).manual_seed(990611)
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    correction_bias = (
        torch.randn(
            (num_experts,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    expected_ids, expected_weights = _sigmoid_bias_topk_reference(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )

    topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_gluon(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_precomputed_topk_fused_route_matches_route_from_topk(num_tokens: int):
    device = "cuda"
    num_experts = 16
    topk = 8
    topk_ids = (
        (torch.arange(num_tokens * topk, device=device, dtype=torch.int32) * 7 + 3)
        .reshape(num_tokens, topk)
        .remainder(num_experts)
    )
    topk_weights = torch.linspace(
        0.125,
        1.0,
        steps=num_tokens * topk,
        device=device,
        dtype=torch.float32,
    ).reshape(num_tokens, topk)

    actual = fused_mxfp_gfx950.gluon_precomputed_topk_fused_route(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=torch.bfloat16,
    )
    expected = fused_mxfp_gfx950._route_from_topk(
        topk_weights,
        topk_ids,
        num_experts,
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    actual_ragged, actual_gather, actual_scatter, actual_gate = actual
    expected_ragged, expected_gather, expected_scatter, expected_gate = expected
    torch.testing.assert_close(actual_gather, expected_gather, rtol=0, atol=0)
    torch.testing.assert_close(actual_scatter, expected_scatter, rtol=0, atol=0)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_ragged.slice_sizes, expected_ragged.slice_sizes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_ragged.slice_offs, expected_ragged.slice_offs, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_ragged.block_offs_data,
        expected_ragged.block_offs_data,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_ragged.block_schedule_data,
        expected_ragged.block_schedule_data,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_package_mxfp4_decode_matches_reference_mfma_exact(num_tokens: int):
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260710 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(router.float(), dim=-1), topk)
    topk_ids = topk_ids.to(torch.int32)

    out = gluon_mxfp4_moe_decode(
        hidden,
        layer.w13_weight_triton_tensor,
        layer.w13_precision_config.b_mx_scale,
        layer.w2_weight_triton_tensor,
        layer.w2_precision_config.b_mx_scale,
        topk_ids,
        topk_weights,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_tokens", [3, 4, 8])
def test_dynamic_mxfp4_dispatch_uses_mfma_with_topk(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    # At M in the precomputed-MFMA decode range the dispatch must route to the
    # MFMA decode kernel and match it bit-exactly. M=3 is the previously-gapped
    # interval (_DIRECT_DECODE_MAX_M, _PRECOMPUTED_MFMA_MIN_M): the apply
    # wrapper now forwards precomputed top-k for it too, so it must also match
    # the MFMA reference exactly rather than falling back to logit routing.
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260710)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(router.float(), dim=-1), topk)
    topk_ids = topk_ids.to(torch.int32)
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    # the reference default runtime path consumes preshuffled Gluon-dot tensors.
    # The tiny-M decode path must accept those same runtime objects directly.
    assert getattr(layer.w13_weight_triton_tensor, "is_shuffled_for_gluon_dot", False)
    assert getattr(layer.w2_weight_triton_tensor, "is_shuffled_for_gluon_dot", False)
    assert not hasattr(layer.w13_weight_triton_tensor, "gluon_decode_clean_weight")
    assert not hasattr(layer.w13_weight_triton_tensor, "gluon_decode_clean_scale")
    assert not hasattr(layer.w2_weight_triton_tensor, "gluon_decode_clean_weight")
    assert not hasattr(layer.w2_weight_triton_tensor, "gluon_decode_clean_scale")

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_dynamic_mxfp4_direct_precomputed_matches_mfma_exact(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260713 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(router.float(), dim=-1), topk)
    topk_ids = topk_ids.to(torch.int32)
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_dynamic_mxfp4_route_owned_default_falls_back_when_direct_unsupported(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden = torch.empty((num_tokens, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((num_tokens, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    intermediate = torch.empty_like(hidden)
    sentinel = torch.empty_like(hidden)
    route_calls = 0
    quantize_calls = 0
    matmul_calls = 0

    def fake_route(*args, **kwargs):
        nonlocal route_calls
        route_calls += 1
        return None, None, None, None

    def fake_quantize(x, *args, **kwargs):
        nonlocal quantize_calls
        quantize_calls += 1
        return x, dummy_scale

    def fake_matmul(*args, **kwargs):
        nonlocal matmul_calls
        matmul_calls += 1
        # gemm1 fuses the intermediate requant (out_quant_format="mxfp4") and
        # returns (intermediate, gemm2_scale); gemm2 returns the final output.
        return (intermediate, dummy_scale) if matmul_calls == 1 else sentinel

    monkeypatch.setattr(fused_mxfp_gfx950, "_dynamic_mxfp4_route", fake_route)
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_quantize_mxfp4_activation",
        fake_quantize,
    )
    monkeypatch.setattr(fused_mxfp_gfx950, "gluon_mxfp_ragged_matmul", fake_matmul)

    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
    )

    assert out is sentinel
    assert route_calls == 1
    # Only the hidden state is quantized explicitly; the intermediate requant is
    # fused into the gemm1 ragged matmul (out_quant_format="mxfp4").
    assert quantize_calls == 1
    assert matmul_calls == 2


def test_dynamic_mxfp4_generic_path_consumes_precomputed_topk(
    monkeypatch: pytest.MonkeyPatch,
):
    """Large-M fallback must not silently recompute routing from logits."""

    hidden = torch.empty((32, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((32, 4), device="cuda", dtype=torch.float32)
    topk_weights = torch.ones((32, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((32, 1), device="cuda", dtype=torch.int32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    intermediate = torch.empty_like(hidden)
    sentinel = torch.empty_like(hidden)
    route_calls = 0
    matmul_calls = 0

    def fake_route_from_topk(weights, ids, *, num_experts, dtype):
        nonlocal route_calls
        route_calls += 1
        assert weights.data_ptr() == topk_weights.data_ptr()
        assert ids.data_ptr() == topk_ids.data_ptr()
        assert num_experts == router.shape[1]
        assert dtype == router.dtype
        return None, None, None, None

    def fail_dynamic_route(*args, **kwargs):
        raise AssertionError("precomputed top-k was ignored")

    def fake_quantize(x, *args, **kwargs):
        return x, dummy_scale

    def fake_matmul(*args, **kwargs):
        nonlocal matmul_calls
        matmul_calls += 1
        # gemm1 fuses the intermediate requant and returns (intermediate, scale).
        return (intermediate, dummy_scale) if matmul_calls == 1 else sentinel

    monkeypatch.setattr(fused_mxfp_gfx950, "_route_from_topk", fake_route_from_topk)
    monkeypatch.setattr(fused_mxfp_gfx950, "_dynamic_mxfp4_route", fail_dynamic_route)
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_quantize_mxfp4_activation",
        fake_quantize,
    )
    monkeypatch.setattr(fused_mxfp_gfx950, "gluon_mxfp_ragged_matmul", fake_matmul)

    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
    )

    assert out is sentinel
    assert route_calls == 1
    assert matmul_calls == 2


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 8])
def test_dynamic_mxfp4_route_owned_softmax_mfma_decode(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260711 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids, topk_weights = invoke_softmax_topk_route_gluon(router, topk)
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 8])
def test_dynamic_mxfp4_route_owned_kimi_sigmoid_mfma_decode(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    scale = 2.827
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260712 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    correction_bias = torch.tensor(
        [0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.05, -0.05],
        device=device,
        dtype=torch.float32,
    )
    topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_gluon(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=scale,
        normalize_topk_weights=True,
    )
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=correction_bias,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=scale,
        normalize_topk_weights=True,
        routing_method_type=2,
        w13_bias=None,
        w2_bias=None,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Precomputed entry point (``gluon_mxfp_precomputed_mxfp4_fused_moe``) dispatch.
#
# This is the entry consumed by ``gluon_mxfp4_precomputed_moe_apply`` (the
# ``routing_mode="precomputed_topk"`` registered kernel). These tests pin down
# its *current* dispatch behavior so we can validate the observation that,
# unlike ``gluon_mxfp_dynamic_mxfp4_fused_moe``, it does NOT route to the decode
# or package-prefill fast paths: every batch size falls through to the generic
# ragged path. If we later wire the fast paths into this entry, these tests are
# the ones that must change.
# ---------------------------------------------------------------------------


def _spy_precomputed_entry_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, int], object]:
    """Install spies on every dispatch helper the precomputed entry could hit.

    Returns ``(counters, sentinel)`` where ``counters`` maps each helper to its
    call count and ``sentinel`` is the object the stubbed ragged path returns.
    The generic ragged path is stubbed so no real kernel launches are needed;
    the decode / package-prefill helpers raise if the precomputed entry ever
    reaches them.
    """

    counters: dict[str, int] = {
        "package_prefill": 0,
        "direct_decode": 0,
        "mfma_decode": 0,
        "fused_route": 0,
        "route_from_topk": 0,
        "ragged": 0,
    }

    def fail_package_prefill(*args, **kwargs):
        counters["package_prefill"] += 1
        raise AssertionError(
            "precomputed entry unexpectedly reached the package-prefill path"
        )

    def fail_direct_decode(*args, **kwargs):
        counters["direct_decode"] += 1
        raise AssertionError(
            "precomputed entry unexpectedly reached the direct MFMA decode path"
        )

    def fail_mfma_decode(*args, **kwargs):
        counters["mfma_decode"] += 1
        raise AssertionError(
            "precomputed entry unexpectedly reached the precomputed-MFMA decode path"
        )

    def fake_fused_route(*args, **kwargs):
        counters["fused_route"] += 1
        return None, None, None, None

    def fake_route_from_topk(*args, **kwargs):
        counters["route_from_topk"] += 1
        return None, None, None, None

    sentinel = object()

    def fake_ragged(*args, **kwargs):
        counters["ragged"] += 1
        return sentinel

    monkeypatch.setattr(
        fused_mxfp_gfx950, "_maybe_gluon_package_mxfp4_prefill", fail_package_prefill
    )
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_direct_mfma_decode",
        fail_direct_decode,
    )
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_mfma_decode",
        fail_mfma_decode,
    )
    monkeypatch.setattr(
        fused_mxfp_gfx950, "gluon_precomputed_topk_fused_route", fake_fused_route
    )
    monkeypatch.setattr(fused_mxfp_gfx950, "_route_from_topk", fake_route_from_topk)
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_gluon_mxfp_dynamic_mxfp4_fused_moe_from_route",
        fake_ragged,
    )
    return counters, sentinel


def _make_dummy_weights_1x4(device: str = "cuda"):
    # Rank-3 expert weight tensor so ``_extract_gluon_raw_w_unshuffled`` accepts
    # it; the ragged path is stubbed so contents never get read.
    w = torch.empty((1, 4, 4), device=device, dtype=torch.uint8)
    scale = torch.empty((1, 4, 1), device=device, dtype=torch.uint8)
    return w, scale


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_precomputed_entry_decode_sizes_use_ragged_not_fast_paths(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Decode-sized batches must NOT reach decode/package-prefill from this entry.

    This documents the gap versus ``gluon_mxfp_dynamic_mxfp4_fused_moe``: the
    precomputed entry ignores the tuned decode kernels and always builds ragged
    metadata + runs the generic matmul.
    """
    counters, sentinel = _spy_precomputed_entry_dispatch(monkeypatch)
    w, scale = _make_dummy_weights_1x4()
    topk_weights = torch.ones((num_tokens, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.int32)
    hidden = torch.empty((num_tokens, 8), device="cuda", dtype=torch.bfloat16)

    out = gluon_mxfp_precomputed_mxfp4_fused_moe(
        hidden,
        topk_weights,
        topk_ids,
        w,
        w,
        w13_mx_scale=scale,
        w2_mx_scale=scale,
    )

    assert out is sentinel
    assert counters["ragged"] == 1
    # The fast paths are never consulted.
    assert counters["package_prefill"] == 0
    assert counters["direct_decode"] == 0
    assert counters["mfma_decode"] == 0
    # Small-M uses the single-kernel fused route; either way we routed exactly
    # once and never recomputed from logits.
    assert counters["fused_route"] + counters["route_from_topk"] == 1


def test_precomputed_entry_prefill_size_uses_ragged_not_package_prefill(
    monkeypatch: pytest.MonkeyPatch,
):
    """Prefill-sized batches must NOT reach the package-prefill path either."""
    counters, sentinel = _spy_precomputed_entry_dispatch(monkeypatch)
    w, scale = _make_dummy_weights_1x4()
    num_tokens = 128
    topk_weights = torch.ones((num_tokens, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.int32)
    hidden = torch.empty((num_tokens, 8), device="cuda", dtype=torch.bfloat16)

    out = gluon_mxfp_precomputed_mxfp4_fused_moe(
        hidden,
        topk_weights,
        topk_ids,
        w,
        w,
        w13_mx_scale=scale,
        w2_mx_scale=scale,
    )

    assert out is sentinel
    assert counters["ragged"] == 1
    assert counters["package_prefill"] == 0
    assert counters["direct_decode"] == 0
    assert counters["mfma_decode"] == 0
    # Large M builds ragged metadata via the generic host route helper.
    assert counters["route_from_topk"] == 1
    assert counters["fused_route"] == 0


def test_precomputed_entry_small_m_uses_fused_route(
    monkeypatch: pytest.MonkeyPatch,
):
    """Small-M precomputed entry uses the single-kernel fused route helper."""
    counters, _sentinel = _spy_precomputed_entry_dispatch(monkeypatch)
    w, scale = _make_dummy_weights_1x4()
    # M=2, top_k=1 -> M < SMALLM_MAX_M and M*top_k <= GLUON_ROUTE_MAX_G.
    topk_weights = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((2, 1), device="cuda", dtype=torch.int32)
    hidden = torch.empty((2, 8), device="cuda", dtype=torch.bfloat16)

    gluon_mxfp_precomputed_mxfp4_fused_moe(
        hidden,
        topk_weights,
        topk_ids,
        w,
        w,
        w13_mx_scale=scale,
        w2_mx_scale=scale,
    )

    assert counters["fused_route"] == 1
    assert counters["route_from_topk"] == 0


def test_precomputed_entry_rejects_missing_or_mismatched_topk():
    """Shape/None validation on the precomputed entry."""
    w, scale = _make_dummy_weights_1x4()
    hidden = torch.empty((2, 8), device="cuda", dtype=torch.bfloat16)

    # rank-1 topk_ids is rejected.
    with pytest.raises(ValueError, match="rank-2"):
        gluon_mxfp_precomputed_mxfp4_fused_moe(
            hidden,
            torch.ones((2,), device="cuda", dtype=torch.float32),
            torch.zeros((2,), device="cuda", dtype=torch.int32),
            w,
            w,
            w13_mx_scale=scale,
            w2_mx_scale=scale,
        )

    # mismatched weights/ids shapes are rejected.
    with pytest.raises(ValueError, match="same shape"):
        gluon_mxfp_precomputed_mxfp4_fused_moe(
            hidden,
            torch.ones((2, 2), device="cuda", dtype=torch.float32),
            torch.zeros((2, 1), device="cuda", dtype=torch.int32),
            w,
            w,
            w13_mx_scale=scale,
            w2_mx_scale=scale,
        )


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 8])
def test_kernel_routing_decode_reaches_route_owned_up_to_decode_max_m(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Kernel-routing (no precomputed top-k) must hit route-owned decode M<=8.

    This pins the dispatch cap lift: without precomputed top-k, batches up to
    ``_DECODE_MAX_M`` must be handed to ``_maybe_route_owned_mxfp4_mfma_decode``
    (which routes in-kernel and dispatches to the tuned decode kernels) rather
    than falling through to the generic ragged GEMM. Previously only M <=
    ``_ROUTE_OWNED_DECODE_MAX_M`` (=2) reached it, so Kimi-style M=8 decode
    silently used the slow ragged path.
    """
    assert num_tokens <= fused_mxfp_gfx950._DECODE_MAX_M
    hidden = torch.empty((num_tokens, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((num_tokens, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    sentinel = torch.empty_like(hidden)
    route_owned_calls = 0
    captured = {}

    def fake_route_owned(*args, **kwargs):
        nonlocal route_owned_calls
        route_owned_calls += 1
        captured["max_m"] = kwargs.get("max_m")
        captured["allow_generic_fallback"] = kwargs.get("allow_generic_fallback")
        return sentinel

    def fail_route(*args, **kwargs):
        raise AssertionError(
            "kernel-routing decode fell through to the ragged reference path"
        )

    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_route_owned_mxfp4_mfma_decode",
        fake_route_owned,
    )
    monkeypatch.setattr(fused_mxfp_gfx950, "_dynamic_mxfp4_route", fail_route)

    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
    )

    assert out is sentinel
    assert route_owned_calls == 1
    # The cap lift forwards the full decode ceiling and lets route-owned reach
    # the precomputed-MFMA decode kernel for the M=3..8 range.
    assert captured["max_m"] == fused_mxfp_gfx950._DECODE_MAX_M
    assert captured["allow_generic_fallback"] is True
