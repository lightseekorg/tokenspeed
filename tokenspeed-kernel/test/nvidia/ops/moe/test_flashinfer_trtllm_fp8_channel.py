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

"""FlashInfer's TRT-LLM-gen FP8 per-channel MoE: weight layout, plan, numerics.

CPU: the resident layout keeps every row's scale with its row through the
gate/up swap, the padding and FlashInfer's row permutations (a wrong scale
shows); the padding adds exactly zero (FP64 emulation of the rounding points);
plan selection. GPU (SM100 to SM103): the kernel against the FP64 emulation.
"""

from __future__ import annotations

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.numerics.torch.activation import torch_silu_and_mul
from tokenspeed_kernel.numerics.torch.moe import torch_moe_experts
from tokenspeed_kernel.numerics.torch.quantization import (
    quantize_fp8_rows,
    quantize_fp8_rows_trtllm,
    round_to_bf16_exact,
)
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from tokenspeed_kernel.platform import Platform, current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError

pytestmark = pytest.mark.skipif(
    find_spec("flashinfer") is None or not current_platform().is_nvidia,
    reason="FlashInfer's TRT-LLM MoE is registered only on NVIDIA",
)

ROUTED = "flashinfer_trtllm_fp8_channel_routed_moe_apply"
KERNEL_ROUTED = "flashinfer_trtllm_fp8_channel_moe_apply"
TRITON = "triton_fp8_channel_precomputed_moe_apply"
FP8 = torch.float8_e4m3fn


def _experts(experts: int, ispp: int, hidden: int, seed: int, device="cpu"):
    """FP8 experts with a distinct FP32 scale per row: w13 gate rows then up
    rows, w2."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def quantized(shape):
        weight = torch.randn(shape, generator=generator, device=device) * 0.02
        scale = weight.abs().amax(dim=-1) / 448.0
        return (weight / scale[..., None]).to(FP8), scale.float()

    w13, w13_scale = quantized((experts, 2 * ispp, hidden))
    w2, w2_scale = quantized((experts, hidden, ispp))
    return w13, w13_scale, w2, w2_scale


def _dequantized(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.double() * scale.double()[..., None]


def _logical(w13, w13_scale, w2, w2_scale, ispp: int):
    """The checkpoint's dequantized gate, up and down of the kernel layout:
    rows put back by the inverse permutations, [up | gate] swapped back,
    padding dropped (and checked zero)."""
    from tokenspeed_kernel.ops.moe.flashinfer import (
        fp8_channel_row_permutations,
    )

    padded = w2.shape[-1]
    zeros13 = w13.new_zeros(w13.shape[1:])
    zeros2 = w2.new_zeros(w2.shape[1:])
    rows13, rows2 = fp8_channel_row_permutations(zeros13, zeros2)
    d13 = _dequantized(w13, w13_scale)[:, torch.argsort(rows13)]
    d2 = _dequantized(w2, w2_scale)[:, torch.argsort(rows2)]
    assert not d13[:, ispp:padded].any() and not d13[:, padded + ispp :].any()
    assert not d2[:, :, ispp:].any()
    up, gate = d13[:, :ispp], d13[:, padded : padded + ispp]
    return torch.cat([gate, up], dim=1), d2[:, :, :ispp]


@pytest.mark.parametrize("ispp", [192, 288, 384, 2048])
def test_layout_keeps_each_scale_with_its_row(ispp):
    from tokenspeed_kernel.ops.moe.flashinfer import (
        TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT,
        prepare_fp8_channel_weights,
    )

    w13, w13_scale, w2, w2_scale = _experts(2, ispp, 256, seed=ispp)
    layout = prepare_fp8_channel_weights(w13, w13_scale, w2, w2_scale)
    alignment = TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT
    padded = -(-ispp // alignment) * alignment
    assert layout[0].shape == (2, 2 * padded, 256) and layout[0].dtype == FP8
    assert layout[1].shape == (2, 2 * padded) and layout[1].dtype == torch.float32
    assert layout[2].shape == (2, 256, padded) and layout[3].shape == (2, 256)
    d13, d2 = _logical(*layout, ispp)
    assert torch.equal(d13, _dequantized(w13, w13_scale))
    assert torch.equal(d2, _dequantized(w2, w2_scale))


@pytest.mark.parametrize("which", ["gate", "up", "down"])
def test_a_wrong_channel_scale_shows(which):
    from tokenspeed_kernel.ops.moe.flashinfer import (
        prepare_fp8_channel_weights,
    )

    ispp = 192
    w13, w13_scale, w2, w2_scale = _experts(2, ispp, 256, seed=1)
    reference = (_dequantized(w13, w13_scale), _dequantized(w2, w2_scale))
    w13_scale, w2_scale = w13_scale.clone(), w2_scale.clone()
    # One channel of expert 1, scaled by 2.
    if which == "gate":
        w13_scale[1, 7] *= 2
    elif which == "up":
        w13_scale[1, ispp + 7] *= 2
    else:
        w2_scale[1, 7] *= 2
    d13, d2 = _logical(*prepare_fp8_channel_weights(w13, w13_scale, w2, w2_scale), ispp)
    differs13 = (d13 != reference[0]).any(dim=-1)
    differs2 = (d2 != reference[1]).any(dim=-1)
    changed = torch.nonzero(torch.cat([differs13, differs2], dim=1)).tolist()
    row = {"gate": 7, "up": ispp + 7, "down": 2 * ispp + 7}[which]
    assert changed == [[1, row]]


def test_scales_left_unpermuted_show():
    from tokenspeed_kernel.ops.moe.flashinfer import (
        TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT,
        fp8_channel_row_permutations,
        pad_fp8_channel_weights,
    )

    ispp = 192
    w13, w13_scale, w2, w2_scale = _experts(1, ispp, 256, seed=2)
    padded = pad_fp8_channel_weights(
        w13, w13_scale, w2, w2_scale, TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT
    )
    rows13, rows2 = fp8_channel_row_permutations(padded[0][0], padded[2][0])
    # The weights' rows permuted, the scales not: the permutation is no
    # identity, so the layout no longer dequantizes to the checkpoint.
    wrong = (
        padded[0].index_select(1, rows13),
        padded[1],
        padded[2].index_select(1, rows2),
        padded[3],
    )
    d13, d2 = _logical(*wrong, ispp)
    assert not torch.equal(d13, _dequantized(w13, w13_scale))
    assert not torch.equal(d2, _dequantized(w2, w2_scale))


def _emulate(
    x,
    w13,
    w13_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    up_first,
    quantized_input=None,
    local_experts=None,
):
    """FP64 emulation of the kernel's rounding points on a layout before the
    row shuffle: the input quantized per token (``quantized_input``: the codes
    and scales the kernel received), FC1 with SiLU(gate) * up in its epilogue
    rounded to BF16, the FC2 input quantized per route over the (padded)
    partition as the kernel does (TRT-LLM's per-token quantization), FC2
    rounded to BF16 per route, the FP32 combine. ``local_experts``: the global
    id of the first of this rank's experts (``w13.shape[0]`` of them); routes
    to other experts add nothing."""
    if quantized_input is None:
        x_codes, x_scale = quantize_fp8_rows(x.double())
    else:
        x_codes, x_scale = (t.cpu() for t in quantized_input)
        x_scale = x_scale.reshape(-1)

    def activation(fc1: torch.Tensor) -> torch.Tensor:
        first, second = fc1.chunk(2, dim=-1)
        up, gate = (first, second) if up_first else (second, first)
        return round_to_bf16_exact(gate * torch.sigmoid(gate) * up)

    def requantize(activated: torch.Tensor) -> torch.Tensor:
        codes, scale = quantize_fp8_rows_trtllm(activated)
        return codes.double() * scale.double()[:, None]

    output = torch_moe_experts(
        x_codes.double() * x_scale.double()[:, None],
        _dequantized(w13, w13_scale),
        _dequantized(w2, w2_scale),
        topk_ids,
        topk_weights,
        activation=activation,
        w13_bias=None,
        w2_bias=None,
        quantize_input=None,
        quantize_intermediate=requantize,
        round_expert_output=round_to_bf16_exact,
        first_expert=local_experts or 0,
    )
    return round_to_bf16_exact(output)


def _reference(
    x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, local_experts=0
):
    """The experts' exact math in FP64 on the dequantized weights, without any
    activation quantization or rounding: what the FP8 scheme approximates."""
    return torch_moe_experts(
        x.double(),
        _dequantized(w13, w13_scale),
        _dequantized(w2, w2_scale),
        topk_ids,
        topk_weights,
        activation=lambda gate_up: torch_silu_and_mul(gate_up, None),
        w13_bias=None,
        w2_bias=None,
        quantize_input=None,
        quantize_intermediate=None,
        round_expert_output=None,
        first_expert=local_experts,
    )


def test_padding_adds_exactly_zero():
    from tokenspeed_kernel.ops.moe.flashinfer import (
        pad_fp8_channel_weights,
    )

    ispp, hidden, experts, top_k = 192, 256, 4, 2
    w13, w13_scale, w2, w2_scale = _experts(experts, ispp, hidden, seed=3)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(16, hidden, generator=generator).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=generator)[:top_k] for _ in range(16)]
    )
    topk_weights = torch.rand(16, top_k, generator=generator)
    # A 192 partition needs no padding at 64; 128 exercises it.
    padded = pad_fp8_channel_weights(w13, w13_scale, w2, w2_scale, 128)
    assert padded[2].shape[-1] == 256
    unpadded = _emulate(x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, False)
    on_layout = _emulate(x, *padded, topk_weights, topk_ids, True)
    assert torch.equal(on_layout, unpadded)


def test_routes_pack_ids_and_bf16_weights():
    from tokenspeed_kernel.ops.moe.flashinfer import _pack_routes

    ids = torch.tensor([[0, 255], [17, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.25], [1.0, 0.375]])
    packed = _pack_routes(ids, weights)
    assert packed.dtype == torch.int32
    assert torch.equal(packed >> 16, ids)
    bits = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    assert torch.equal(bits.float(), weights)


def _stub_vendor_calls(monkeypatch):
    """Replace the input quantizer and both vendor MoE calls with stubs that
    record the routing arguments; returns the list of recorded calls."""
    import tokenspeed_kernel.ops.moe.flashinfer as module

    calls = []

    def vendor(name):
        def call(**kwargs):
            calls.append((name, kwargs))
            tokens, hidden = kwargs["hidden_states"].shape
            out = torch.zeros(tokens * kwargs["top_k"], hidden, dtype=torch.bfloat16)
            expanded = torch.arange(tokens * kwargs["top_k"], dtype=torch.int32)
            return out, None, expanded

        return call

    monkeypatch.setattr(
        module,
        "quantize_fp8_with_scale",
        lambda x, granularity: (x.to(FP8), x.new_ones(x.shape[0])),
    )
    monkeypatch.setattr(module, "trtllm_fp8_per_channel_scale_moe", vendor("logits"))
    monkeypatch.setattr(
        module, "trtllm_fp8_per_channel_scale_routed_moe", vendor("routed")
    )
    return calls


def _stub_experts(top_k=2):
    w = SimpleNamespace(
        _spec=SimpleNamespace(
            num_experts=8, top_k=top_k, ep_rank=0, num_local_experts=8
        ),
        fp8_channel_output_scales=torch.nn.Parameter(torch.ones(8)),
        intermediate_size_per_partition=64,
    )
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        setattr(w, name, torch.nn.Parameter(torch.empty(0)))
    return w


def test_kernel_routing_variant_runs_supplied_routes(monkeypatch):
    """Precomputed routes passed to the kernel-routing registration run on
    those routes, not on routes recomputed from the router logits."""
    from tokenspeed_kernel.ops.moe.flashinfer import (
        _pack_routes,
        flashinfer_trtllm_fp8_channel_moe_apply,
    )

    calls = _stub_vendor_calls(monkeypatch)
    x = torch.ones(3, 128, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 5], [7, 1], [2, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.25], [1.0, 0.375], [0.125, 0.75]])
    _, weights, _ = flashinfer_trtllm_fp8_channel_moe_apply(
        {},
        x,
        _stub_experts(),
        torch.zeros(3, 8),
        topk_weights,
        topk_ids,
        do_finalize=False,
    )
    assert [name for name, _ in calls] == ["routed"]
    assert torch.equal(calls[0][1]["topk_ids"], _pack_routes(topk_ids, topk_weights))
    assert weights is topk_weights
    with pytest.raises(ValueError, match="both topk_weights and topk_ids"):
        flashinfer_trtllm_fp8_channel_moe_apply(
            {}, x, _stub_experts(), torch.zeros(3, 8), None, topk_ids
        )


@pytest.mark.parametrize("kernel", [KERNEL_ROUTED, ROUTED])
def test_zero_tokens_keep_the_route_weight_dtype(kernel, monkeypatch):
    """A rank with no tokens hands back the caller's route weights, so a
    deferred finalize sees the same dtype as on non-empty batches."""
    import tokenspeed_kernel.ops.moe.flashinfer as module

    calls = _stub_vendor_calls(monkeypatch)
    apply = getattr(module, kernel)
    x = torch.empty(0, 128, dtype=torch.bfloat16)
    topk_weights = torch.empty(0, 2, dtype=torch.float32)
    topk_ids = torch.empty(0, 2, dtype=torch.int32)
    output, weights, expanded = apply(
        {},
        x,
        _stub_experts(),
        torch.empty(0, 8),
        topk_weights,
        topk_ids,
        do_finalize=False,
    )
    assert calls == []
    assert output.shape == (0, 128) and expanded.shape == (0,)
    assert weights is topk_weights
    if kernel == KERNEL_ROUTED:
        _, weights, _ = apply(
            {}, x, _stub_experts(), torch.empty(0, 8), do_finalize=False
        )
        assert weights.shape == (0, 2) and weights.dtype == torch.bfloat16


def _plan_on(platform, **overrides) -> dict:
    kwargs = dict(
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=192,
        hidden=4096,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )
    kwargs.update(overrides)
    registry = KernelRegistry.get()
    real_platform = Platform.get()
    try:
        Platform.override(platform)
        registry.clear_cache()
        return kernel_moe_plan("fp8_channel", **kwargs)
    finally:
        Platform.override(real_platform)
        registry.clear_cache()


@pytest.mark.parametrize("platform_fixture", ["b200_platform", "b300_platform"])
@pytest.mark.parametrize("ispp,ep_size", [(192, 1), (2048, 1), (2048, 8)])
def test_blackwell_plans_the_vendor_kernel(platform_fixture, ispp, ep_size, request):
    platform = request.getfixturevalue(platform_fixture)
    for solution in (None, "flashinfer_trtllm"):
        plan = _plan_on(platform, ispp=ispp, ep_size=ep_size, solution=solution)
        assert plan["apply_kernel_name"] == ROUTED
        assert plan["weight_preprocessor"] is not None
        assert plan["supports_precomputed_topk"] and not plan["support_routing"]
    plan = _plan_on(platform, routing_mode="kernel_routing")
    assert plan["apply_kernel_name"] == KERNEL_ROUTED and plan["support_routing"]
    # The Triton kernel stays reachable by solution.
    assert _plan_on(platform, solution="triton")["apply_kernel_name"] == TRITON


def test_hopper_keeps_the_triton_kernel(h100_platform):
    assert _plan_on(h100_platform)["apply_kernel_name"] == TRITON
    with pytest.raises(NoKernelFoundError):
        _plan_on(h100_platform, solution="flashinfer_trtllm")


# ---------------------------------------------------------------------------
# GPU: the kernel against the FP64 emulation.
# ---------------------------------------------------------------------------

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the per-channel FP8 MoE cubins are SM100 / SM103",
)


def _vendor_experts(w13, w13_scale, w2, w2_scale, num_experts, top_k, ep_rank):
    """The routed kernel's module: this rank's experts in the kernel layout."""
    from tokenspeed_kernel.ops.moe.flashinfer import (
        flashinfer_trtllm_fp8_channel_moe_process_weights,
    )

    w = torch.nn.Module()
    for name, value in zip(
        ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
        (w13, w13_scale, w2, w2_scale),
    ):
        w.register_parameter(
            name, torch.nn.Parameter(value.cuda(), requires_grad=False)
        )
    w._spec = SimpleNamespace(
        num_experts=num_experts,
        top_k=top_k,
        ep_rank=ep_rank,
        num_local_experts=w13.shape[0],
    )
    flashinfer_trtllm_fp8_channel_moe_process_weights({}, w)
    return w


def _routes(tokens, experts, top_k, weights_dtype, generator):
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=generator)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, top_k, generator=generator).to(weights_dtype)
    return topk_ids, topk_weights


@requires_sm100
@pytest.mark.parametrize("tokens", [1, 8, 64, 512])
# 320 and 2,560: Beam's partitions at TP8 and under --ep-size 8.
@pytest.mark.parametrize("ispp", [192, 2048, 320, 2560])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ep", [1, 4])
def test_routed_kernel_matches_the_emulation(tokens, ispp, weights_dtype, ep):
    """The routed kernel (rank 1 of ``ep``: a quarter of the experts under EP)
    against the FP64 emulation of its rounding points and the exact math.

    The kernel is held (a) within 1e-2 relative L2 of the emulation, which
    follows its FC2-input rule including the BF16 step before E4M3, and (b)
    as accurate as the emulation against the exact math (15 % and 1e-3 slack
    on the scheme's own error). A wrong row scale fails both by orders of
    magnitude."""
    from tokenspeed_kernel.ops.moe.flashinfer import (
        flashinfer_trtllm_fp8_channel_routed_moe_apply,
    )
    from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

    experts, hidden, top_k = 16, 1024, 4
    ep_rank = 1 if ep > 1 else 0
    local = experts // ep
    first = ep_rank * local
    w13, w13_scale, w2, w2_scale = _experts(local, ispp, hidden, seed=tokens)
    generator = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, hidden, generator=generator).to(torch.bfloat16)
    topk_ids, topk_weights = _routes(tokens, experts, top_k, weights_dtype, generator)
    quantized = quantize_fp8_with_scale(x.cuda(), granularity="token")
    route_weights = topk_weights.float()
    expected = _emulate(
        x,
        w13,
        w13_scale,
        w2,
        w2_scale,
        route_weights,
        topk_ids,
        False,
        quantized_input=quantized,
        local_experts=first,
    )
    exact = _reference(x, w13, w13_scale, w2, w2_scale, route_weights, topk_ids, first)

    w = _vendor_experts(w13, w13_scale, w2, w2_scale, experts, top_k, ep_rank)
    out = flashinfer_trtllm_fp8_channel_routed_moe_apply(
        {}, x.cuda(), w, None, topk_weights.cuda(), topk_ids.cuda()
    )
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    out = out.double().cpu()

    def rel(a, b):
        return float((a - b).norm() / b.norm())

    if not exact.any():  # no route reached this rank's experts
        assert not out.any()
        return
    scheme = rel(expected, exact)
    assert rel(out, expected) < 1e-2, (rel(out, expected), scheme)
    assert rel(out, exact) <= 1.15 * scheme + 1e-3, (rel(out, exact), scheme)


@requires_sm100
@pytest.mark.parametrize("tokens", [1, 64, 512])
@pytest.mark.parametrize("ispp", [192, 320])
def test_unpadded_partition_equals_the_padded_one(tokens, ispp, monkeypatch):
    """A partition that is a multiple of 64 but not of 128 (192, and Beam's
    320 at TP8) runs unpadded: bit-identical to the same experts padded to
    the next multiple of 128."""
    import tokenspeed_kernel.ops.moe.flashinfer as module

    experts, hidden, top_k = 16, 1024, 4
    w13, w13_scale, w2, w2_scale = _experts(experts, ispp, hidden, seed=tokens)
    generator = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, hidden, generator=generator).to(torch.bfloat16).cuda()
    topk_ids, topk_weights = _routes(tokens, experts, top_k, torch.float32, generator)
    outputs = []
    for alignment in (module.TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT, 128):
        monkeypatch.setattr(module, "TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT", alignment)
        w = _vendor_experts(w13, w13_scale, w2, w2_scale, experts, top_k, 0)
        padded = ispp if alignment == 64 else -(-ispp // 128) * 128
        assert w.intermediate_size_per_partition == padded
        outputs.append(
            module.flashinfer_trtllm_fp8_channel_routed_moe_apply(
                {}, x, w, None, topk_weights.cuda(), topk_ids.cuda()
            )
        )
    assert torch.equal(outputs[0], outputs[1])
