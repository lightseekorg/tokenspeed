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

"""CPU contracts of the BF16x3 FP32-weight GEMM: the weight split, a CPU
model of the kernel's arithmetic against FP64, the tile and split-K choice,
and the route through ``decode_gemv``."""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest
import tokenspeed_kernel.ops.gemm  # noqa: F401  (registration side effects)
import torch
from tokenspeed_kernel.ops.gemm import triton_bf16x3, triton_gemv
from tokenspeed_kernel.ops.gemm.triton_bf16x3 import (
    BF16X3_MIN_M,
    _bf16x3_config,
    split_fp32_weight_bf16x3,
    triton_bf16x3_gemm_fp32,
)
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, Priority

U = 2.0**-24  # FP32 unit roundoff
BF16_MAX = torch.finfo(torch.bfloat16).max


def _fp32(bits: torch.Tensor) -> torch.Tensor:
    """FP32 values from bit patterns held in int64."""
    return (
        torch.where(bits >= 2**31, bits - 2**32, bits)
        .to(torch.int32)
        .view(torch.float32)
    )


def _random_fp32(n: int, min_exponent: int, max_exponent: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    exponent = torch.randint(min_exponent, max_exponent + 1, (n,), generator=g)
    mantissa = torch.randint(0, 1 << 23, (n,), generator=g)
    sign = torch.randint(0, 2, (n,), generator=g)
    return _fp32((sign << 31) | ((exponent + 127) << 23) | mantissa)


def test_split_is_exact_for_weights_from_2_pow_minus_110():
    edges = torch.tensor([0.0, -0.0, 2.0**-110, -(2.0**-126), 1.0, BF16_MAX, -BF16_MAX])
    # Finite values that round to infinity in BF16, up to the largest FP32.
    past_bf16 = _fp32(torch.tensor([0x7F7F8000, 0x7F7FFFFF, 0xFF7FC000]))
    w = torch.cat([_random_fp32(1 << 16, -110, 127, seed=0), edges, past_bf16])
    pieces = split_fp32_weight_bf16x3(w.view(1, -1))
    assert pieces.dtype == torch.bfloat16 and pieces.shape == (3, 1, w.numel())
    assert pieces.is_contiguous()
    assert torch.isfinite(pieces).all()
    assert torch.equal(pieces.double().sum(0).view(-1), w.double())


def test_split_of_tiny_weights_stays_within_half_the_bf16_subnormal_spacing():
    # FP32 subnormals and normals below 2^-110: the last piece rounds to the
    # BF16 subnormal grid, 2^-133.
    subnormal = _fp32(
        torch.randint(1, 1 << 23, (4096,), generator=torch.Generator().manual_seed(1))
    )
    w = torch.cat([subnormal, -subnormal, _random_fp32(4096, -126, -111, seed=2)])
    pieces = split_fp32_weight_bf16x3(w.view(2, -1))
    error = (pieces.double().sum(0).view(-1) - w.double()).abs()
    assert error.max().item() <= 2.0**-134
    assert error.max().item() > 0  # the bound is reached, not vacuous


def test_split_keeps_non_finite_weights_in_the_leading_piece():
    w = torch.tensor([[math.inf, -math.inf, math.nan, 3.0]])
    pieces = split_fp32_weight_bf16x3(w)
    assert torch.equal(pieces[0].float().isinf(), w.isinf())
    assert pieces[0].float()[0, :2].tolist() == [math.inf, -math.inf]
    assert pieces[0].float()[0, 2].isnan()
    assert (pieces[1:, :, :3] == 0).all()


def test_split_rejects_other_layouts():
    with pytest.raises(ValueError, match="float32"):
        split_fp32_weight_bf16x3(torch.zeros(4, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="float32"):
        split_fp32_weight_bf16x3(torch.zeros(4, 8, 2))


def _rz32(v: torch.Tensor) -> torch.Tensor:
    """FP64 to FP32 rounded toward zero."""
    f = v.to(torch.float32)
    over = f.double().abs() > v.abs()
    return torch.where(over, torch.nextafter(f, torch.zeros_like(f)), f)


def _block_rn(x, w):
    """One MMA accumulator span as the exact sum rounded once to FP32."""
    return sum(x @ w[p].t() for p in range(3)).to(torch.float32)


def _block_rz(x, w):
    """One MMA accumulator span as an FP32 accumulator truncated after every
    16 products of a piece, small pieces first: a pessimistic stand-in for the
    tensor core's adder."""
    part = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32)
    for k0 in range(0, x.shape[1], 16):
        for p in (2, 1, 0):
            part = _rz32(part.double() + x[:, k0 : k0 + 16] @ w[p, :, k0 : k0 + 16].t())
    return part


def _kernel_model(x, pieces, split, block, drain=64):
    """The kernel's arithmetic on CPU: exact BF16 products; per split, one MMA
    accumulator span per ``drain`` elements of K, each added to an FP32
    register accumulator in K order; split partials added in split order."""
    xd, wd = x.double(), pieces.double()
    k = x.shape[1]
    ks = k // split
    out = None
    for s in range(split):
        acc = torch.zeros(x.shape[0], pieces.shape[1], dtype=torch.float32)
        for k0 in range(s * ks, (s + 1) * ks, drain):
            acc = acc + block(xd[:, k0 : k0 + drain], wd[:, :, k0 : k0 + drain])
        out = acc if out is None else out + acc
    return out


def _error_units(y, x, w):
    """Largest |y - x.w| against FP64, in units of 2^-24 sum|x w|."""
    exact = x.double() @ w.double().t()
    magnitude = x.double().abs() @ w.double().abs().t()
    return ((y.double() - exact).abs() / (U * magnitude)).max().item()


def _model_problem():
    g = torch.Generator().manual_seed(3)
    m, n, k = 24, 128, 1024
    x = torch.randn(m, k, generator=g).to(torch.bfloat16)
    w = torch.randn(n, k, generator=g) * 0.02
    return x, w


# These tests check the arithmetic the kernel is built on, with a CPU model;
# they do not run the Triton kernel. Its GPU tests check the kernel itself.
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize(
    "block", [_block_rn, _block_rz], ids=["round_once", "truncate_per_step"]
)
def test_kernel_model_with_the_drain_stays_within_two_units(split, block):
    """With every product exact and the MMA accumulator drained every 64
    elements of K, even the truncating stand-in stays within 2 units of
    2^-24 sum|x w| (0.84 here; 0.45 when each block rounds once)."""
    x, w = _model_problem()
    y = _kernel_model(x, split_fp32_weight_bf16x3(w), split, block)
    assert y.dtype == torch.float32
    assert _error_units(y, x, w) <= 2


def test_kernel_model_without_the_drain_leaves_the_bound():
    """One truncating MMA accumulator over all of K: 12 units here. Dropping a
    weight piece costs 7, so the two-unit bound above also needs all three."""
    x, w = _model_problem()
    pieces = split_fp32_weight_bf16x3(w)
    assert _error_units(_kernel_model(x, pieces, 1, _block_rz, x.shape[1]), x, w) > 2
    pieces[2] = 0
    assert _error_units(_kernel_model(x, pieces, 1, _block_rz), x, w) > 2


@pytest.mark.parametrize(
    "block", [_block_rn, _block_rz], ids=["round_once", "truncate_per_step"]
)
def test_kernel_model_returns_one_hot_rows_exactly(block):
    """A row with a single 1 picks one column of w: each partial sum of its
    pieces is an FP32 value, so even a truncating adder returns it exactly."""
    g = torch.Generator().manual_seed(5)
    n, k = 128, 1024
    w = torch.randn(n, k, generator=g) * 0.02
    cols = torch.randint(0, k, (24,), generator=g)
    x = torch.zeros(24, k, dtype=torch.bfloat16)
    x[torch.arange(24), cols] = 1
    y = _kernel_model(x, split_fp32_weight_bf16x3(w), 4, block)
    assert torch.equal(y, w[:, cols].t())


def test_kernel_model_passes_non_finite_values_like_an_fp32_gemm():
    x = torch.ones(3, 64)
    x[1, 3] = 0.0  # 0 * inf
    x[2] = -1.0
    w = torch.ones(4, 64)
    w[0, 3] = math.inf
    w[1, 5] = math.nan
    w[2, 3], w[2, 4] = -math.inf, math.inf
    x = x.to(torch.bfloat16)
    y = _kernel_model(x, split_fp32_weight_bf16x3(w), 1, _block_rn)
    expected = x.float() @ w.t()
    assert torch.equal(y.isnan(), expected.isnan())
    assert torch.equal(y.isinf(), expected.isinf())
    assert torch.equal(y[expected.isinf()], expected[expected.isinf()])


def test_kernel_model_keeps_non_finite_activations_non_finite():
    """An infinite activation meets pieces of either sign or zero, so outputs
    that an FP32 GEMM gives as +-inf can be NaN; none of them is finite."""
    g = torch.Generator().manual_seed(6)
    x = torch.randn(3, 64, generator=g).to(torch.bfloat16)
    x[0, 5] = math.inf
    x[1, 7] = math.nan
    w = torch.randn(32, 64, generator=g) * 0.02
    y = _kernel_model(x, split_fp32_weight_bf16x3(w), 1, _block_rn)
    expected = x.float() @ w.t()
    assert torch.equal(y.isfinite(), expected.isfinite())
    assert expected[0].isinf().all() and y[0].isnan().any()


@pytest.mark.parametrize("n,k", [(128, 2048), (256, 4096), (256, 7168), (1024, 3072)])
def test_config_tiles_the_problem_with_a_bounded_set_of_kernels(n, k):
    blocks = k // 64
    variants, splits = set(), set()
    for m in [*range(BF16X3_MIN_M, 1100), 2048, 4095, 8192, 16384]:
        cfg = _bf16x3_config(m, n, k)
        weight_tile = cfg.block_a if cfg.swap else cfg.block_b
        row_tile = cfg.block_b if cfg.swap else cfg.block_a
        assert n % weight_tile == 0 and blocks % cfg.split == 0
        assert row_tile & (row_tile - 1) == 0 and (not cfg.swap or row_tile >= m)
        ctas = (n // weight_tile) * -(-m // row_tile) * cfg.split
        assert ctas >= (120 if m <= 512 else 240) or cfg.split == blocks
        variants.add((cfg.swap, cfg.block_a, cfg.block_b, cfg.split > 1))
        splits.add(cfg.split)
    # Each variant and split is one compile, whatever the row count.
    assert len(variants) <= 5 and len(splits) <= 8


class _Launches(list):
    """Stands in for the Triton kernels and records each launch."""

    def kernel(self, name):
        launches = self

        class _Kernel:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: launches.append(
                    (name, grid, args, kwargs)
                )

        return _Kernel()


@pytest.mark.parametrize("m", [17, 64, 65, 600, 4096])
def test_kernel_is_launched_with_the_drain_and_without_fp_fusion(monkeypatch, m):
    """The bounds above hold for an accumulator drained every 64 elements of K
    and adds that are not fused; the launch must ask for both. This runs the
    host side only, on CPU."""
    launches = _Launches()
    monkeypatch.setattr(triton_bf16x3, "_bf16x3_gemm_kernel", launches.kernel("gemm"))
    monkeypatch.setattr(
        triton_bf16x3, "_bf16x3_splitk_reduce_kernel", launches.kernel("reduce")
    )
    n, k = 256, 4096
    cfg = _bf16x3_config(m, n, k)
    out = triton_bf16x3_gemm_fp32(
        torch.zeros(m, k, dtype=torch.bfloat16),
        torch.zeros(3, n, k, dtype=torch.bfloat16),
    )
    assert out.shape == (m, n) and out.dtype == torch.float32
    name, grid, args, kwargs = launches[0]
    # One CTA per output tile and split.
    if cfg.swap:
        tiles = (n // kwargs["BA"], -(-m // kwargs["BB"]))
    else:
        tiles = (-(-m // kwargs["BA"]), n // kwargs["BB"])
    assert name == "gemm" and grid == (*tiles, cfg.split)
    assert kwargs["BK"] == kwargs["DRAIN"] == 64
    assert kwargs["enable_fp_fusion"] is False
    assert kwargs["SWAP"] == cfg.swap and kwargs["USE_WS"] == (cfg.split > 1)
    assert args[4:] == (m, k // cfg.split)
    if cfg.split == 1:
        assert len(launches) == 1 and args[2] is out
        return
    workspace = args[2]
    assert workspace is not out and workspace.numel() == cfg.split * m * n
    name, grid, args, kwargs = launches[1]
    assert name == "reduce" and len(launches) == 2
    assert args[0] is workspace and args[1] is out and args[2] == m * n
    assert grid == (-(-m * n // kwargs["BLOCK"]),)
    assert kwargs["SPLIT"] == cfg.split and kwargs["enable_fp_fusion"] is False


@pytest.fixture
def select_on(monkeypatch):
    def use(platform):
        monkeypatch.setattr(triton_gemv, "current_platform", lambda: platform)
        triton_gemv._select.cache_clear()

    yield use
    triton_gemv._select.cache_clear()


def _selected(m, n, k, weight_split=True):
    return triton_gemv._select(
        m, n, k, True, torch.bfloat16, torch.float32, weight_split
    )


def test_registry_selects_the_split_kernel_from_17_rows(select_on, b200_platform):
    select_on(b200_platform)
    for m in (17, 64, 65, 1000, 8192):
        assert _selected(m, 256, 4096) is triton_bf16x3_gemm_fp32
    for m in (1, 8, 16):
        assert _selected(m, 256, 4096) is not triton_bf16x3_gemm_fp32
    # Without the split, or outside the registered shapes, other kernels run.
    assert _selected(64, 256, 4096, weight_split=False) is not triton_bf16x3_gemm_fp32
    for n, k in ((320, 4096), (2048, 4096), (256, 4000)):
        assert _selected(64, n, k) is not triton_bf16x3_gemm_fp32


def test_registry_keeps_the_split_kernel_to_sm100(
    select_on, h100_platform, mi350_platform
):
    for platform in (h100_platform, mi350_platform):
        select_on(platform)
        assert _selected(64, 256, 4096) is not triton_bf16x3_gemm_fp32


def _cuda_weight(n, k, dtype=torch.float32):
    """A contiguous CUDA weight as decode_gemv_weight_split() reads it."""
    weight = Mock(is_cuda=True, ndim=2, shape=(n, k), dtype=dtype)
    weight.is_contiguous.return_value = True
    return weight


def test_weight_split_follows_the_split_kernel_registration(
    monkeypatch, select_on, b200_platform, h100_platform, mi350_platform
):
    split = Mock(return_value=object())
    monkeypatch.setattr(triton_gemv, "split_fp32_weight_bf16x3", split)
    select_on(b200_platform)
    weight = _cuda_weight(256, 4096)
    assert triton_gemv.decode_gemv_weight_split(weight) is split.return_value
    split.assert_called_once_with(weight)
    for n, k in ((320, 4096), (2048, 4096), (256, 4000)):
        assert triton_gemv.decode_gemv_weight_split(_cuda_weight(n, k)) is None
    for dtype in (torch.bfloat16, torch.float16):
        weight = _cuda_weight(256, 4096, dtype)
        assert triton_gemv.decode_gemv_weight_split(weight) is None
    for platform in (h100_platform, mi350_platform):
        select_on(platform)
        assert triton_gemv.decode_gemv_weight_split(_cuda_weight(256, 4096)) is None
    assert split.call_count == 1


def test_weights_of_128_rows_stay_off_the_split_kernel(
    monkeypatch, select_on, b200_platform
):
    """On GB200 the FP32 Torch product beat this kernel for 128-row weights at
    most row counts up to 128, so the registration starts at 256 weight rows:
    a 128-row weight gets neither the kernel nor a split."""
    split = Mock(return_value=object())
    monkeypatch.setattr(triton_gemv, "split_fp32_weight_bf16x3", split)
    select_on(b200_platform)
    for m in (17, 64, 128, 1024, 8192):
        assert _selected(m, 128, 4096) is not triton_bf16x3_gemm_fp32
    assert triton_gemv.decode_gemv_weight_split(_cuda_weight(128, 4096)) is None
    split.assert_not_called()
    for n in (256, 384, 1024):
        for m in (17, 128, 8192):
            assert _selected(m, n, 4096) is triton_bf16x3_gemm_fp32
        weight = _cuda_weight(n, 4096)
        assert triton_gemv.decode_gemv_weight_split(weight) is split.return_value
        split.assert_called_with(weight)
    assert split.call_count == 3


def test_k_outside_the_measured_2048_to_8192_stays_off_the_split_kernel(
    monkeypatch, select_on, b200_platform
):
    """The GB200 sweep covered K from 2048 to 8192, so the registration stops
    there: other K get neither the kernel nor a split, the ends get both."""
    split = Mock(return_value=object())
    monkeypatch.setattr(triton_gemv, "split_fp32_weight_bf16x3", split)
    select_on(b200_platform)
    for k in (1024, 1984, 8256, 16384):
        for m in (17, 128, 8192):
            assert _selected(m, 256, k) is not triton_bf16x3_gemm_fp32
        assert triton_gemv.decode_gemv_weight_split(_cuda_weight(256, k)) is None
    split.assert_not_called()
    for k in (2048, 8192):
        for m in (17, 128, 8192):
            assert _selected(m, 256, k) is triton_bf16x3_gemm_fp32
        weight = _cuda_weight(256, k)
        assert triton_gemv.decode_gemv_weight_split(weight) is split.return_value
        split.assert_called_with(weight)
    assert split.call_count == 2


def test_weight_split_is_made_under_a_kernel_preferred_for_some_rows(
    monkeypatch, select_on, b200_platform
):
    """A kernel registered ahead of the split kernel for rows 17 to 96 serves
    those rows from the weight; the split is still made for the rows above."""
    stand_in = Mock()
    reg = KernelRegistry.get()
    reg.register(
        KernelSpec(
            name="test_decode_gemv_rows_17_96",
            family="gemm",
            mode="decode_gemv",
            solution="test",
            format_signatures=triton_gemv._BF16_FP32_SIG,
            traits={"m_min": frozenset({17}), "m_max": frozenset({96})},
            priority=Priority.SPECIALIZED + 1,
        ),
        stand_in,
    )
    try:
        split = Mock(return_value=object())
        monkeypatch.setattr(triton_gemv, "split_fp32_weight_bf16x3", split)
        select_on(b200_platform)
        weight = _cuda_weight(256, 4096)
        assert triton_gemv.decode_gemv_weight_split(weight) is split.return_value
        split.assert_called_once_with(weight)
        for m in range(17, 97):
            assert _selected(m, 256, 4096) is stand_in
        for m in (97, 128, 1024, 8192):
            assert _selected(m, 256, 4096) is triton_bf16x3_gemm_fp32
        for m in (1, 16):
            assert _selected(m, 256, 4096) not in (stand_in, triton_bf16x3_gemm_fp32)
    finally:
        reg._unregister("test_decode_gemv_rows_17_96")


def test_decode_gemv_hands_the_split_to_the_split_kernel(monkeypatch):
    result = torch.empty(32, 128)
    kernel = Mock(return_value=result)
    select = Mock(return_value=kernel)
    monkeypatch.setattr(triton_gemv, "triton_bf16x3_gemm_fp32", kernel)
    monkeypatch.setattr(triton_gemv, "_select", select)
    x = torch.zeros(32, 256, dtype=torch.bfloat16)
    w = torch.zeros(128, 256)
    pieces = torch.zeros(3, 128, 256, dtype=torch.bfloat16)
    assert triton_gemv.decode_gemv(x, w, weight_split=pieces) is result
    kernel.assert_called_once_with(x, pieces, None)
    assert select.call_args.args[-3:] == (torch.bfloat16, torch.float32, True)


def test_decode_gemv_widens_bf16_rows_for_an_fp32_weight_on_torch():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(20, 64, generator=g).to(torch.bfloat16)
    w = torch.randn(32, 64, generator=g)
    out = torch.empty(20, 32)
    expected = x.float() @ w.t()
    assert torch.equal(triton_gemv.decode_gemv(x, w), expected)
    assert triton_gemv.decode_gemv(x, w, out=out) is out
    assert torch.equal(out, expected)
    # On CPU the split is not used; the result is the same Torch product.
    pieces = split_fp32_weight_bf16x3(w)
    assert torch.equal(triton_gemv.decode_gemv(x, w, weight_split=pieces), expected)
    assert triton_gemv.decode_gemv_weight_split(w) is None
    with pytest.raises(ValueError, match="out must match"):
        triton_gemv.decode_gemv(x, w, out=torch.empty(20, 32, dtype=torch.bfloat16))
