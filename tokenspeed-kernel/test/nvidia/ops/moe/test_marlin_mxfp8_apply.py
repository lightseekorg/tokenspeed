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

"""Numerical check: Marlin W8A16 MXFP8 MoE apply vs a bf16 dequant reference.

E4M3 times an E8M0 power of two is exact in bf16, so the Marlin output must
track bf16 GEMMs over the dequantized experts. Runs on SM90+.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe.marlin import mxfp8
from tokenspeed_kernel.ops.moe.marlin.mxfp8 import (
    dequantize_marlin_mxfp8,
    marlin_mxfp8_moe_weights,
    marlin_mxfp8_precomputed_moe_apply,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import moe_align_block_size

BLOCK = 32


def _requires_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA required")
    if current_platform().arch_version < type(current_platform().arch_version)(9, 0):
        pytest.skip("SM90+ required")


def _make_weights(num_experts, hidden, ispp, generator, scale_range=(118, 124)):
    """Random E4M3 experts with raw E8M0 (uint8) scales per 1x32 block."""

    def fp8(*shape):
        values = torch.randn(*shape, generator=generator, device="cuda") * 2
        return values.clamp(-448, 448).to(torch.float8_e4m3fn)

    def e8m0(*shape):
        lo, hi = scale_range
        return torch.randint(
            lo, hi + 1, shape, generator=generator, device="cuda", dtype=torch.uint8
        )

    return {
        "w13_weight": fp8(num_experts, 2 * ispp, hidden),
        "w13_weight_scale_inv": e8m0(num_experts, 2 * ispp, hidden // BLOCK),
        "w2_weight": fp8(num_experts, hidden, ispp),
        "w2_weight_scale_inv": e8m0(num_experts, hidden, ispp // BLOCK),
    }


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = torch.exp2(scale.float() - 127).repeat_interleave(BLOCK, dim=-1)
    return (weight.float() * s).to(torch.bfloat16)


def _reference(x, raw, topk_ids, topk_weights, limit):
    """bf16 GEMMs over dequantized experts with the clamped SwiGLU epilogue."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(raw["w13_weight"].shape[0]):
        token_ids, slot_ids = (topk_ids == expert).nonzero(as_tuple=True)
        if not token_ids.numel():
            continue
        w13 = _dequant(raw["w13_weight"][expert], raw["w13_weight_scale_inv"][expert])
        w2 = _dequant(raw["w2_weight"][expert], raw["w2_weight_scale_inv"][expert])
        gate_up = (x[token_ids].float() @ w13.float().T).to(torch.bfloat16)
        gate, up = gate_up.float().chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        down = (hidden.float() @ w2.float().T).to(torch.bfloat16)
        out.index_add_(
            0, token_ids, down.float() * topk_weights[token_ids, slot_ids][:, None]
        )
    return out


class _Experts(torch.nn.Module):
    def __init__(self, raw, ep_size, ep_rank, limit):
        super().__init__()
        for name, value in raw.items():
            setattr(self, name, torch.nn.Parameter(value, requires_grad=False))
        self.num_local_experts = raw["w13_weight"].shape[0]
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.tp_size = 1
        self.tp_rank = 0
        self.activation = "swiglu"
        self.swiglu_arg = SimpleNamespace(alpha=None, limit=limit)
        self.swiglu_beta = None
        self._moe_backend_state = None


def _routing(num_tokens, num_experts, top_k, generator):
    scores = torch.rand(num_tokens, num_experts, generator=generator, device="cuda")
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights.float()


def _run(raw, x, topk_ids, topk_weights, ep_size, limit):
    num_experts = raw["w13_weight"].shape[0]
    num_local = num_experts // ep_size
    plan = {"activation": "swiglu", "deterministic_finalize": True}
    acc = torch.zeros_like(x, dtype=torch.float32)
    for ep_rank in range(ep_size):
        lo = ep_rank * num_local
        shard = {k: v[lo : lo + num_local].clone() for k, v in raw.items()}
        w = _Experts(shard, ep_size, ep_rank, limit)
        marlin_mxfp8_moe_weights(plan, w)
        acc += marlin_mxfp8_precomputed_moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        ).float()
    return acc


def _assert_close(actual, expected):
    magnitude = expected.abs().max().item()
    torch.testing.assert_close(actual, expected, atol=2e-2 * magnitude, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 300])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_marlin_mxfp8_matches_dequant_reference(num_tokens: int, ep_size: int):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(7)
    num_experts, top_k, hidden, ispp = 8, 3, 512, 256
    raw = _make_weights(num_experts, hidden, ispp, generator)
    x = (torch.randn(num_tokens, hidden, generator=generator, device="cuda")).to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)

    expected = _reference(x, raw, topk_ids, topk_weights, limit=None)
    actual = _run(raw, x, topk_ids, topk_weights, ep_size, limit=None)
    _assert_close(actual, expected)


@pytest.mark.parametrize("num_tokens", [4, 32])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_marlin_mxfp8_honors_swiglu_clamp(num_tokens: int, ep_size: int):
    """The clamp must change the answer by far more than the tolerance, so an
    epilogue that drops it fails instead of passing by accident. 4 tokens take
    the GEMV kernels, 32 the Marlin GEMMs."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(13)
    num_experts, top_k, hidden, ispp = 8, 2, 256, 128
    limit = 3.0
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(124, 126))
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)

    expected = _reference(x, raw, topk_ids, topk_weights, limit)
    unclamped = _reference(x, raw, topk_ids, topk_weights, None)
    assert (expected - unclamped).abs().max() > 0.5 * expected.abs().max()
    actual = _run(raw, x, topk_ids, topk_weights, ep_size, limit)
    _assert_close(actual, expected)


@pytest.mark.parametrize("num_tokens", [2, 300])
def test_marlin_mxfp8_without_ep_skips_invalid_and_repeated_ids(num_tokens: int):
    """Without EP, ``-1`` slots contribute zero and a repeated id counts once
    per slot. Two tokens fit the GEMV route budget but stay on Marlin without
    EP. 300 tokens (1200 routes) outgrow the small-batch alignment kernel;
    both batches stay below the dense threshold, so the Marlin GEMMs see the
    -1 blocks."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(37)
    num_experts, top_k, hidden, ispp = 8, 4, 512, 256
    assert num_tokens * top_k / num_experts < mxfp8._DENSE_MIN_ROUTES_PER_EXPERT
    raw = _make_weights(num_experts, hidden, ispp, generator)
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
    topk_ids[:, 1] = topk_ids[:, 0]
    topk_ids[::2, 2] = -1
    topk_ids[1] = -1

    expected = _reference(x, raw, topk_ids, topk_weights, limit=None)
    actual = _run(raw, x, topk_ids, topk_weights, ep_size=1, limit=None)
    _assert_close(actual, expected)
    assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize("num_experts,ep_size", [(16, 1), (300, 1), (64, 8), (1100, 1)])
def test_deterministic_align_is_reproducible(num_experts: int, ep_size: int):
    """The deterministic alignment is bitwise stable across calls (EP-masked ids
    included), matches the atomic scatter up to the order inside a segment,
    and past the small-batch kernel (>= 1024 routes or > 64 expert lanes)
    lists every expert's routes in route order. Global ids aligned against
    the rank's expert_start lay out exactly like pre-remapped local ids."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(5)
    num_local = num_experts // ep_size
    expert_start = num_local if ep_size > 1 else 0
    for num_tokens in (100, 300, 4096):
        topk_ids, _ = _routing(num_tokens, num_experts, 8, generator)
        topk_ids[0, 0] = -1
        local = topk_ids - expert_start
        local = torch.where((local >= 0) & (local < num_local), local, -1)
        local = local.to(torch.int32)

        def align(ids, start, deterministic):
            return moe_align_block_size(
                ids, 16, num_local, expert_start=start, deterministic=deterministic
            )

        stable = align(local, 0, True)
        total = int(stable[2].item())
        for ids, start in ((local, 0), (local, 0), (topk_ids, expert_start)):
            again = align(ids, start, True)
            assert int(again[2].item()) == total
            assert torch.equal(again[0][:total], stable[0][:total])
            assert torch.equal(again[1][: total // 16], stable[1][: total // 16])
        fast = align(topk_ids, expert_start, False)
        assert total == int(fast[2].item())
        assert torch.equal(stable[1][: total // 16], fast[1][: total // 16])
        sorted_ids = stable[0][:total]
        numel = local.numel()
        if numel >= 1024 or num_local + 1 > 64:
            # Each block holds one expert's routes followed by padding.
            blocks = sorted_ids.view(-1, 16)
            valid = blocks < numel
            assert torch.all(valid[:, :-1] | ~valid[:, 1:])
            assert torch.all((blocks[:, 1:] > blocks[:, :-1]) | ~valid[:, 1:])
        assert torch.equal(
            sorted_ids[sorted_ids < numel].sort().values,
            torch.arange(numel, device="cuda", dtype=torch.int32),
        )
        assert torch.equal(
            stable[0][:total].sort().values, fast[0][:total].sort().values
        )


def test_marlin_mxfp8_is_bitwise_reproducible():
    """deterministic_finalize: repeated applies on one batch are bitwise equal,
    including batches whose routes exceed the small-batch alignment kernel and
    the prefill-sized one that takes the dense path."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(17)
    num_experts, top_k, hidden, ispp = 16, 4, 1024, 512
    raw = _make_weights(num_experts, hidden, ispp, generator)
    w = _Experts({k: v.clone() for k, v in raw.items()}, 1, 0, 7.0)
    plan = {"activation": "swiglu", "deterministic_finalize": True}
    marlin_mxfp8_moe_weights(plan, w)
    for num_tokens in (5, 256, 512, 2048):
        x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
            torch.bfloat16
        )
        topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
        first = marlin_mxfp8_precomputed_moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        )
        for _ in range(3):
            again = marlin_mxfp8_precomputed_moe_apply(
                plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
            )
            assert torch.equal(first, again)


@pytest.mark.parametrize("num_tokens", [4, 1024])
def test_marlin_mxfp8_production_geometry(num_tokens: int):
    """hidden 6144 / ispp 2048 / top-8 over 256 experts, one EP32 rank's eight
    local experts; routes to other ranks must contribute nothing. 4 tokens
    (32 routes) sit above the 16-route GEMV gate, so both sizes stay on
    Marlin."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(19)
    num_global, ep_size, top_k, hidden, ispp = 256, 32, 8, 6144, 2048
    num_local = num_global // ep_size
    ep_rank = 5
    raw = _make_weights(num_local, hidden, ispp, generator)
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_global, top_k, generator)
    # Route token 0 to this rank so even the 4-token batch exercises the GEMMs.
    topk_ids[0, 0] = ep_rank * num_local

    local_ids = topk_ids - ep_rank * num_local
    local_ids = torch.where((local_ids >= 0) & (local_ids < num_local), local_ids, -1)
    expected = _reference(x, raw, local_ids, topk_weights, limit=7.0)

    w = _Experts(raw, ep_size, ep_rank, 7.0)
    plan = {"activation": "swiglu", "deterministic_finalize": True}
    marlin_mxfp8_moe_weights(plan, w)
    actual = marlin_mxfp8_precomputed_moe_apply(
        plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
    ).float()
    _assert_close(actual, expected)
    untouched = (local_ids < 0).all(dim=1)
    assert torch.count_nonzero(actual[untouched]) == 0


@pytest.mark.parametrize("num_experts,hidden,ispp", [(3, 256, 128), (2, 6144, 2048)])
def test_marlin_dequant_inverts_the_repack(num_experts: int, hidden: int, ispp: int):
    """Dequantizing the Marlin layout reproduces the bf16 experts bitwise, with
    w13's gate/up halves swapped into the CUTLASS order."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(23)
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(100, 160))
    w = _Experts({k: v.clone() for k, v in raw.items()}, 1, 0, None)
    marlin_mxfp8_moe_weights(
        {"activation": "swiglu", "deterministic_finalize": True}, w
    )

    w13 = torch.empty(
        num_experts, 2 * ispp, hidden, dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.empty(num_experts, hidden, ispp, dtype=torch.bfloat16, device="cuda")
    dequantize_marlin_mxfp8(w.w13_weight, w.w13_weight_scale_inv, w13, swap_halves=True)
    dequantize_marlin_mxfp8(w.w2_weight, w.w2_weight_scale_inv, w2, swap_halves=False)

    expected_w13 = _dequant(raw["w13_weight"], raw["w13_weight_scale_inv"])
    assert torch.equal(w13, torch.cat(expected_w13.chunk(2, dim=1)[::-1], dim=1))
    assert torch.equal(w2, _dequant(raw["w2_weight"], raw["w2_weight_scale_inv"]))


@pytest.mark.parametrize("limit", [None, 3.0])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_dense_prefill_path_matches_reference(
    ep_size: int, limit: float | None, monkeypatch
):
    """Prefill-sized batches take the dequant + CUTLASS bf16 path; it must match
    the reference and the Marlin path it replaces, ``-1`` slots included."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(29)
    num_experts, top_k, hidden, ispp, num_tokens = 16, 4, 512, 256, 1024
    assert num_tokens * top_k / num_experts >= mxfp8._DENSE_MIN_ROUTES_PER_EXPERT
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(124, 126))
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
    topk_ids[::3, 1] = -1

    expected = _reference(x, raw, topk_ids, topk_weights, limit)
    dense = _run(raw, x, topk_ids, topk_weights, ep_size, limit)
    _assert_close(dense, expected)
    monkeypatch.setattr(mxfp8, "_DENSE_MIN_ROUTES_PER_EXPERT", float("inf"))
    marlin = _run(raw, x, topk_ids, topk_weights, ep_size, limit)
    _assert_close(dense, marlin)


def test_dense_prefill_path_shares_the_bf16_cutlass_call(monkeypatch):
    """The dense path calls CUTLASS as the BF16 CUTLASS apply does: PDL off
    even when the caller asks for it, on that apply's persistent workspace."""
    _requires_sm90()
    from tokenspeed_kernel.ops.moe.flashinfer import cutlass_unquant

    calls = []

    def fake_cutlass_fused_moe(**kwargs):
        calls.append(kwargs)
        return [torch.zeros_like(kwargs["input"])]

    monkeypatch.setattr(cutlass_unquant, "cutlass_fused_moe", fake_cutlass_fused_moe)
    generator = torch.Generator(device="cuda").manual_seed(31)
    num_experts, top_k, hidden, ispp, num_tokens = 16, 4, 512, 256, 1024
    raw = _make_weights(num_experts, hidden, ispp, generator)
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
    plan = {"activation": "swiglu", "deterministic_finalize": False}
    w = _Experts(raw, 1, 0, 3.0)
    marlin_mxfp8_moe_weights(plan, w)
    marlin_mxfp8_precomputed_moe_apply(
        plan,
        x,
        w,
        None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        enable_pdl=True,
    )

    (call,) = calls
    assert call["enable_pdl"] is False
    assert call["workspace_buffer"] is cutlass_unquant._workspaces[x.device]
    assert call["use_fused_finalize"] is True
    assert call["swiglu_limit"] is w._moe_backend_state.cutlass_swiglu_limit


@pytest.mark.parametrize("num_tokens", [1, 2])
@pytest.mark.parametrize("ispp", [512, 192])
def test_gemv_path_matches_reference_and_marlin(
    num_tokens: int, ispp: int, monkeypatch
):
    """Decode-sized batches under EP take the GEMV kernels; they must match
    the reference and the Marlin GEMMs they replace, with ``-1`` slots and
    other ranks' experts. At ispp 192 the down projection's K holds three
    GEMV steps, so it runs as a single K range."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(41)
    num_experts, top_k, hidden, limit = 16, 4, 1024, 7.0
    assert num_tokens * top_k <= mxfp8._GEMV_MAX_ROUTES
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(122, 126))
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    # Rank 0 owns [0, 8): one local route per token, the rest sit on rank 1.
    topk_ids = torch.tensor(
        [[0, 8, 9, -1], [1, 10, 11, 12]][:num_tokens],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.rand((num_tokens, top_k), generator=generator, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    expected = _reference(x, raw, topk_ids, topk_weights, limit)
    gemv = _run(raw, x, topk_ids, topk_weights, 2, limit)
    _assert_close(gemv, expected)
    monkeypatch.setattr(mxfp8, "_GEMV_MAX_ROUTES", 0)
    marlin = _run(raw, x, topk_ids, topk_weights, 2, limit)
    _assert_close(gemv, marlin)


def test_prefill_stays_on_marlin_when_dequant_cannot_tile_ispp():
    """ispp 192 meets Marlin's alignment but not the dequant's K width, so a
    prefill-sized batch must keep the Marlin GEMMs."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(31)
    num_experts, top_k, hidden, ispp, num_tokens = 8, 4, 256, 192, 512
    assert num_tokens * top_k / num_experts >= mxfp8._DENSE_MIN_ROUTES_PER_EXPERT
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(124, 126))
    w = _Experts({k: v.clone() for k, v in raw.items()}, 1, 0, None)
    plan = {"activation": "swiglu", "deterministic_finalize": True}
    marlin_mxfp8_moe_weights(plan, w)
    assert not w._moe_backend_state.dense_prefill
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
    actual = marlin_mxfp8_precomputed_moe_apply(
        plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
    ).float()
    _assert_close(actual, _reference(x, raw, topk_ids, topk_weights, limit=None))


def test_gemv_is_faster_is_host_only():
    """No GPU read: decode CUDA graphs capture this apply."""
    assert not mxfp8._gemv_is_faster(8, ep_size=1)
    assert not mxfp8._gemv_is_faster(24, ep_size=32)
    assert mxfp8._gemv_is_faster(8, ep_size=32)
    assert mxfp8._gemv_is_faster(16, ep_size=32)


def test_marlin_mxfp8_rejects_malformed_scales():
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(3)
    raw = _make_weights(2, 256, 128, generator)
    raw["w2_weight_scale_inv"] = raw["w2_weight_scale_inv"][..., :-1].contiguous()
    w = _Experts(raw, 1, 0, None)
    with pytest.raises(ValueError, match="1x32 blocks"):
        marlin_mxfp8_moe_weights(
            {"activation": "swiglu", "deterministic_finalize": True}, w
        )


@pytest.mark.parametrize("num_tokens", [2, 32, 1024])
def test_marlin_mxfp8_strided_inputs_match_contiguous(num_tokens):
    """Strided FP32 routing and K-strided activations on GEMV, Marlin, CUTLASS."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(47)
    hidden, ispp, experts, top_k = 256, 128, 8, 2
    raw = _make_weights(experts, hidden, ispp, generator)
    x_storage = torch.randn(
        num_tokens, 2 * hidden, device="cuda", generator=generator
    ).bfloat16()
    x = x_storage[:, ::2]
    ids, weights = _routing(num_tokens, experts, top_k, generator)
    weight_storage = torch.full((num_tokens, 2 * top_k), 99.0, device="cuda")
    weight_storage[:, ::2] = weights
    strided_weights = weight_storage[:, ::2]
    assert not x.is_contiguous() and not strided_weights.is_contiguous()
    expected = _run(raw, x.contiguous(), ids, weights, ep_size=2, limit=None)
    actual = _run(raw, x, ids, strided_weights, ep_size=2, limit=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("code", [0, 255])
@pytest.mark.parametrize("name", ["w13_weight_scale_inv", "w2_weight_scale_inv"])
def test_marlin_mxfp8_rejects_scale_encoding_before_repack(monkeypatch, code, name):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(53)
    raw = _make_weights(2, 256, 128, generator)
    raw[name].view(-1)[0] = code
    w = _Experts(raw, 1, 0, None)
    original = {name: parameter for name, parameter in w.named_parameters()}

    def unexpected_repack(*args, **kwargs):
        pytest.fail("invalid scales reached weight repacking")

    monkeypatch.setattr(mxfp8, "gptq_marlin_repack", unexpected_repack)
    with pytest.raises(ValueError, match="unsupported E8M0"):
        marlin_mxfp8_moe_weights(
            {"activation": "swiglu", "deterministic_finalize": True}, w
        )
    assert w._moe_backend_state is None
    for name, parameter in w.named_parameters():
        assert parameter is original[name]


def test_marlin_mxfp8_accepts_finite_normal_scale_endpoints():
    mxfp8._validate_mxfp8_scales(
        "scales", torch.tensor([1, 127, 254], dtype=torch.uint8)
    )


@pytest.mark.parametrize("size_n", [64, 128, 192, 320])
@pytest.mark.parametrize("swap_halves", [False, True])
def test_dequant_half_swap_wraps_each_row(size_n, swap_halves):
    """Odd N-tile counts wrap row by row without overwriting output guards."""
    _requires_sm90()
    size_k = 128
    # Enough space to contain the old kernel's entire erroneous extra tile.
    guard = 64 * size_k
    # Each Marlin 16x64 tile has one constant E4M3 value. Distinct N tiles
    # also check the half-swap permutation, not merely that every row was set.
    values = torch.arange(1, size_n // 64 + 1, device="cuda").float()
    codes = values.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    words = codes * 0x01010101
    qweight = torch.empty(
        (1, size_k // 16, size_n * 4), dtype=torch.int32, device="cuda"
    )
    qweight.view(1, size_k // 16, size_n // 64, 256).copy_(words.view(1, 1, -1, 1))
    scales = torch.full(
        (1, size_k // BLOCK, size_n), 127, dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)
    storage = torch.full(
        (size_n * size_k + 2 * guard,),
        -17.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = storage[guard:-guard].view(1, size_n, size_k)
    dequantize_marlin_mxfp8(qweight, scales, out, swap_halves=swap_halves)
    expected_rows = values.repeat_interleave(64)
    if swap_halves:
        expected_rows = expected_rows.roll(size_n // 2)
    expected = expected_rows.bfloat16().view(1, size_n, 1).expand_as(out)
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
    assert torch.all(storage[:guard] == -17)
    assert torch.all(storage[-guard:] == -17)


@pytest.mark.parametrize(
    "case,exception,message",
    [
        ("weight_dtype", TypeError, "qweight must be int32"),
        ("scale_dtype", TypeError, "scales must be float8_e8m0fnu"),
        ("output_dtype", TypeError, "out must have bf16"),
        ("weight_rank", ValueError, "three-dimensional"),
        ("weight_stride", ValueError, "contiguous"),
        ("scale_stride", ValueError, "contiguous"),
        ("output_stride", ValueError, "contiguous"),
        ("weight_shape", ValueError, "Marlin weights"),
        ("scale_shape", ValueError, "Marlin scales"),
        ("alignment", ValueError, "dequant needs"),
        ("empty", ValueError, "must be positive"),
        ("weight_device", ValueError, "share a device"),
        ("scale_device", ValueError, "share a device"),
        ("cpu", ValueError, "requires CUDA"),
    ],
)
def test_dequant_rejects_invalid_storage_before_launch(
    monkeypatch, case, exception, message
):
    """Metadata-only rejection never reads storage or launches a GPU kernel."""
    qweight = torch.empty(1, 8, 512, dtype=torch.int32)
    scales = torch.empty(1, 4, 128, dtype=torch.float8_e8m0fnu)
    out = torch.empty(1, 128, 128, dtype=torch.bfloat16)
    if case == "weight_dtype":
        # The right logical shape with one quarter of the required bytes.
        qweight = torch.empty(qweight.shape, dtype=torch.uint8)
    elif case == "scale_dtype":
        scales = torch.empty(scales.shape, dtype=torch.float32)
    elif case == "output_dtype":
        out = out.float()
    elif case == "weight_rank":
        qweight = qweight.flatten()
    elif case == "weight_stride":
        qweight = torch.empty(1, 16, 512, dtype=qweight.dtype)[:, ::2, :]
    elif case == "scale_stride":
        scales = torch.empty(1, 8, 128, dtype=scales.dtype)[:, ::2, :]
    elif case == "output_stride":
        out = torch.empty(1, 256, 128, dtype=out.dtype)[:, ::2, :]
    elif case == "weight_shape":
        qweight = qweight[:, :7, :]
    elif case == "scale_shape":
        scales = scales[:, :3, :]
    elif case == "alignment":
        qweight = torch.empty(1, 8, 384, dtype=qweight.dtype)
        scales = torch.empty(1, 4, 96, dtype=scales.dtype)
        out = torch.empty(1, 96, 128, dtype=out.dtype)
    elif case == "empty":
        qweight, scales, out = qweight[:0], scales[:0], out[:0]
    elif case == "weight_device":
        qweight = torch.empty(qweight.shape, dtype=qweight.dtype, device="meta")
    elif case == "scale_device":
        scales = torch.empty(scales.shape, dtype=scales.dtype, device="meta")

    class ForbiddenLaunch:
        def __getitem__(self, grid):
            pytest.fail(f"invalid {case} input reached kernel launch: {grid}")

    monkeypatch.setattr(mxfp8, "_marlin_mxfp8_to_bf16_kernel", ForbiddenLaunch())
    with pytest.raises(exception, match=message):
        mxfp8.dequantize_marlin_mxfp8(qweight, scales, out, swap_halves=True)
