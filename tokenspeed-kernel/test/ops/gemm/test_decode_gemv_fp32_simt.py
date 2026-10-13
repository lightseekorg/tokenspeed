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

"""Contracts of the FP32 CUDA-core decode GEMM for 17 to 96 rows.

On the CPU, an exact emulation of the kernel's summation order stays inside
the forward error bound of that order, and the registry selects the kernel
only inside its traits: BF16 activations with an FP32 weight and at most 256
outputs; the launch tile follows the timed bands. Where the BF16x3 split
kernel also takes the weight, this kernel keeps rows 17 to 96 and the split
kernel the rows above. On NVIDIA GPUs the kernel equals the emulation bit for
bit, for any row count and tile, and the inputs left to Torch take Torch.
"""

from __future__ import annotations

import random
from fractions import Fraction
from unittest.mock import Mock

import pytest
import tokenspeed_kernel.ops.gemm  # noqa: F401  (registration side effects)
import torch
from tokenspeed_kernel.ops.gemm import gluon_gemv, triton_gemv
from tokenspeed_kernel.ops.gemm.gluon_gemv import gluon_simt_gemm_fp32
from tokenspeed_kernel.ops.gemm.triton_bf16x3 import triton_bf16x3_gemm_fp32
from tokenspeed_kernel.ops.gemm.triton_gemv import (
    _BF16_FP32_SIG,
    _FP32_SIG,
    _select,
    decode_gemv,
    decode_gemv_weight_split,
    torch_decode_gemv,
)
from tokenspeed_kernel.platform import (
    CapabilityRequirement,
    Platform,
    current_platform,
)
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel

_NAME = "gluon_simt_gemm_fp32"
_SPEC = KernelRegistry.get().get_by_name(_NAME)
# The registered (x, weight) dtypes, and the FP32 pair left to Torch.
_MIXED = (torch.bfloat16, torch.float32)
_FP32 = (torch.float32, torch.float32)
_requires_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="the FP32 CUDA-core GEMM is registered for NVIDIA",
)
_requires_split_kernel = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_nvidia
    or not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3),
    reason="the BF16x3 split kernel is registered for SM100 to SM103",
)


def _fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """FP32 ``fma(a, b, c)`` exactly.

    ``a * b`` of two FP32 values is exact in FP64; TwoSum gives the tail of
    ``p + c``, and the FP32 rounding of the FP64 sum is the exact sum's
    except on an FP32 midpoint, where the tail's sign decides.
    """
    p = a.double() * b.double()
    c64 = c.double()
    s = p + c64
    bb = s - p
    tail = (p - (s - bb)) + (c64 - bb)
    r = s.float()
    up = torch.nextafter(r, torch.full_like(r, float("inf")))
    down = torch.nextafter(r, torch.full_like(r, float("-inf")))
    above = (s == (r.double() + up.double()) * 0.5) & (tail > 0)
    below = (s == (r.double() + down.double()) * 0.5) & (tail < 0)
    return torch.where(above, up, torch.where(below, down, r))


def _order_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The kernel's summation order (module doc of gluon_gemv) on the CPU."""
    x, weight = x.float().cpu(), weight.float().cpu()
    (m, k), n = x.shape, weight.shape[0]
    chunk = k // 4
    total = torch.zeros(m, n)
    for s in range(4):
        acc = torch.zeros(m, n, 128)
        for i in range(chunk // 128):
            k0 = s * chunk + 128 * i
            acc = _fma32(
                x[:, None, k0 : k0 + 128].expand(m, n, 128),
                weight[None, :, k0 : k0 + 128].expand(m, n, 128),
                acc,
            )
        # Lane 4 t + v: each warp lane t adds its four chains in order, then
        # the warp runs the xor butterfly.
        lanes = acc.view(m, n, 32, 4)
        u = ((lanes[..., 0] + lanes[..., 1]) + lanes[..., 2]) + lanes[..., 3]
        for offset in (16, 8, 4, 2, 1):
            u = u + u[..., torch.arange(32) ^ offset]
        total = total + u[..., 0]
    return total


def _error_bound(
    x: torch.Tensor, weight: torch.Tensor, rounds: int | None = None
) -> torch.Tensor:
    """``gamma_n * sum |x w|`` for ``n`` roundings per product, plus the FP64
    reference's own error and an underflow allowance.

    ``n`` defaults to the kernel's ``K / 512 + 11``; ``n = K`` holds for an
    FP32 dot product summed in any order.
    """
    k = x.shape[1]
    if rounds is None:
        rounds = k // 512 + 11
    gamma = rounds * 2.0**-24 / (1 - rounds * 2.0**-24)
    magnitude = x.double().abs().cpu() @ weight.double().abs().cpu().t()
    return (gamma + k * 2.0**-52) * magnitude + rounds * 2.0**-149


def _round_to_fp32(value: Fraction) -> float:
    """The FP32 nearest to an exact rational, ties to even (finite range)."""
    if value == 0:
        return 0.0
    size = abs(value)
    exponent = size.numerator.bit_length() - size.denominator.bit_length()
    if Fraction(2) ** exponent > size:
        exponent -= 1
    quantum = Fraction(2) ** (max(exponent, -126) - 23)
    scaled = size / quantum
    rounded = scaled.numerator // scaled.denominator
    rest = scaled - rounded
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and rounded % 2):
        rounded += 1
    return float(rounded * quantum) * (1 if value > 0 else -1)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().cpu().view(torch.int32)


def test_reference_fma_rounds_exactly():
    rng = random.Random(0)
    # 1.5 * (1 + 2^-23) is an FP32 midpoint; a tiny addend must decide it.
    cases = [(1.5, 1 + 2.0**-23, c) for c in (0.0, 2.0**-80, -(2.0**-80))]
    cases += [(1.0, 1.0, 2.0**-24), (-3.0, 2.0**-130, 2.0**-149)]
    for _ in range(3000):
        a = torch.tensor(rng.gauss(0, 1)).bfloat16().float().item()
        b = torch.tensor(rng.gauss(0, 1) * 2 ** rng.randint(-30, 30)).item()
        c = torch.tensor(-a * b * (1 + rng.gauss(0, 2.0**-20))).item()
        cases.append((a, b, c))
    a, b, c = (torch.tensor(column) for column in zip(*cases))
    got = _fma32(a, b, c)
    for (ai, bi, ci), gi in zip(cases, got.tolist()):
        exact = Fraction(ai) * Fraction(bi) + Fraction(ci)
        assert gi == _round_to_fp32(exact), (ai, bi, ci)


@pytest.mark.parametrize("k", [512, 7168])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
def test_reference_stays_inside_the_error_bound(k, x_dtype):
    generator = torch.Generator().manual_seed(k)
    x = torch.randn(17, k, generator=generator).to(x_dtype)
    weight = torch.randn(8, k, generator=generator) * 0.02
    # Rows that cancel: the second half negates the first half's products.
    x[1, k // 2 :] = -x[1, : k // 2]
    weight[1, k // 2 :] = weight[1, : k // 2]
    got = _order_reference(x, weight).double()
    exact = x.double() @ weight.double().t()
    assert ((got - exact).abs() <= _error_bound(x, weight)).all()
    assert (got != exact).any()  # FP32 rounding, not a wider sum


@pytest.fixture
def nvidia_platform(monkeypatch, b200_platform):
    # Swap the singleton itself: Platform.get() would detect the host device.
    monkeypatch.setattr(Platform, "_instance", b200_platform)
    _select.cache_clear()
    yield
    _select.cache_clear()


@pytest.mark.parametrize("dtypes", [_MIXED, _FP32])
@pytest.mark.parametrize(
    "m,n,k,selected",
    [
        (17, 256, 7168, True),
        (96, 128, 4096, True),
        (40, 256, 7168, True),
        (48, 64, 8192, True),
        (32, 4, 512, True),
        (16, 256, 7168, False),
        (97, 256, 7168, False),
        (64, 260, 4096, False),  # more than 256 outputs
        (96, 512, 3072, False),
        (64, 1024, 512, False),
        (64, 254, 4096, False),  # outputs not a multiple of 4
        (64, 256, 4352, False),  # K not a multiple of 512
        (64, 256, 8704, False),  # K past the unrolled bound
        (64, 256, 0, False),
    ],
)
@pytest.mark.parametrize("weight_split", [False, True])
def test_selection_keeps_the_traits(
    nvidia_platform, dtypes, m, n, k, selected, weight_split
):
    # FP32 activations are left to Torch at every shape. The kernel reads the
    # FP32 weight, so a BF16 split of it does not change the selection.
    impl = _select(m, n, k, True, *dtypes, weight_split)
    assert (impl is gluon_simt_gemm_fp32) is (selected and dtypes == _MIXED)
    assert _select(m, n, k, False, *dtypes, weight_split) is not gluon_simt_gemm_fp32


@pytest.mark.parametrize(
    "dtypes",
    [(torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16)],
)
def test_selection_ignores_other_dtypes(nvidia_platform, dtypes):
    assert _select(64, 256, 7168, True, *dtypes, False) is not gluon_simt_gemm_fp32


def test_selection_is_nvidia_only(monkeypatch, mi350_platform):
    monkeypatch.setattr(Platform, "_instance", mi350_platform)
    _select.cache_clear()
    try:
        assert _select(64, 256, 7168, True, *_MIXED, False) is not (
            gluon_simt_gemm_fp32
        )
    finally:
        _select.cache_clear()


def test_band_start_fp32_kernel_keeps_the_other_rows(nvidia_platform, fresh_registry):
    # A broader FP32 kernel at the band start, registered before this one (at
    # equal priority the first registered wins), takes the rows outside 17 to
    # 96 and none inside, and every row of FP32 activations.
    @register_kernel(
        "gemm",
        "decode_gemv",
        name="band_start_fp32_gemm",
        solution="triton",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=_FP32_SIG | _BF16_FP32_SIG,
        traits={"m_min": frozenset({1})},
        priority=Priority.SPECIALIZED,
    )
    def band_start_fp32_gemm(x, weight, out=None):
        raise AssertionError("not launched")

    KernelRegistry.get().register(_SPEC, gluon_simt_gemm_fp32)
    _select.cache_clear()
    for m, expected in (
        (1, band_start_fp32_gemm),
        (16, band_start_fp32_gemm),
        (17, gluon_simt_gemm_fp32),
        (96, gluon_simt_gemm_fp32),
        (97, band_start_fp32_gemm),
        (1024, band_start_fp32_gemm),
    ):
        assert _select(m, 256, 7168, True, *_MIXED, False) is expected
        assert _select(m, 256, 7168, True, *_FP32, False) is band_start_fp32_gemm


def _cuda_weight(n: int, k: int) -> Mock:
    """A contiguous FP32 CUDA weight as decode_gemv_weight_split() reads it."""
    weight = Mock(is_cuda=True, ndim=2, shape=(n, k), dtype=torch.float32)
    weight.is_contiguous.return_value = True
    return weight


def test_split_kernel_keeps_the_rows_above_96(nvidia_platform, monkeypatch):
    # Both FP32-weight kernels as registered. They share weights of 256 rows
    # with K a multiple of 512 from 2048 to 8192: the split is still made for
    # them, this kernel serves rows 17 to 96 from the FP32 weight whether or
    # not the split is passed, and the split kernel serves rows from 97.
    split = Mock(return_value=object())
    monkeypatch.setattr(triton_gemv, "split_fp32_weight_bf16x3", split)
    both = (gluon_simt_gemm_fp32, triton_bf16x3_gemm_fp32)
    shared = range(2048, 8192 + 1, 512)
    for k in shared:
        weight = _cuda_weight(256, k)
        assert decode_gemv_weight_split(weight) is split.return_value
        split.assert_called_with(weight)
        for weight_split in (False, True):
            for m in range(17, 97):
                assert _select(m, 256, k, True, *_MIXED, weight_split) is both[0]
            # Rows up to 16 are left to other kernels.
            for m in (1, 16):
                assert _select(m, 256, k, True, *_MIXED, weight_split) not in both
        for m in (97, 128, 1024, 8192):
            assert _select(m, 256, k, True, *_MIXED, True) is both[1]
            assert _select(m, 256, k, True, *_MIXED, False) is torch_decode_gemv
    assert split.call_count == len(shared)
    # A weight that only one of them takes: 128 rows get no split and Torch
    # from 97 rows; 384 rows get the split kernel from 17 rows.
    assert decode_gemv_weight_split(_cuda_weight(128, 4096)) is None
    assert split.call_count == len(shared)
    for m, expected in ((17, both[0]), (96, both[0]), (97, torch_decode_gemv)):
        assert _select(m, 128, 4096, True, *_MIXED, False) is expected
    assert decode_gemv_weight_split(_cuda_weight(384, 7168)) is split.return_value
    for m in (17, 96, 97):
        assert _select(m, 384, 7168, True, *_MIXED, True) is both[1]
        assert _select(m, 384, 7168, True, *_MIXED, False) is torch_decode_gemv


def test_decode_gemv_hands_this_kernel_the_fp32_weight(monkeypatch):
    # The split goes only to the split kernel; this kernel reads the weight.
    result = torch.empty(32, 256)
    kernel = Mock(return_value=result)
    monkeypatch.setattr(triton_gemv, "_select", Mock(return_value=kernel))
    x = torch.zeros(32, 512, dtype=torch.bfloat16)
    weight = torch.zeros(256, 512)
    pieces = torch.zeros(3, 256, 512, dtype=torch.bfloat16)
    assert decode_gemv(x, weight, weight_split=pieces) is result
    kernel.assert_called_once_with(x, weight, None)


_BF16 = torch.bfloat16


@pytest.mark.parametrize(
    "m,n,k,x_dtype,tile",
    [
        # Up to 128 outputs: [8, 2] below K 6144, [4, 4] from there.
        (17, 4, 512, _BF16, (8, 2, 1, 1)),
        (96, 128, 4096, _BF16, (8, 2, 1, 1)),
        (48, 128, 5632, _BF16, (8, 2, 1, 1)),
        (17, 128, 6144, _BF16, (4, 4, 1, 1)),
        (96, 64, 8192, _BF16, (4, 4, 1, 1)),
        # More outputs below K 4608: [4, 4].
        (17, 132, 512, _BF16, (4, 4, 1, 1)),
        (32, 256, 3584, _BF16, (4, 4, 1, 1)),
        (96, 256, 4096, _BF16, (4, 4, 1, 1)),
        # K 4608 to 5632: by rows.
        (24, 256, 4608, _BF16, (8, 2, 1, 1)),
        (25, 256, 4608, _BF16, (8, 4, 1, 1)),
        (32, 256, 5632, _BF16, (8, 4, 1, 1)),
        (33, 256, 5632, _BF16, (4, 4, 1, 1)),
        (48, 256, 4608, _BF16, (4, 4, 1, 1)),
        (49, 256, 5632, _BF16, (8, 4, 1, 1)),
        (96, 256, 4608, _BF16, (8, 4, 1, 1)),
        # From K 6144: [8, 4] instead of [8, 2] up to 24 rows.
        (24, 256, 6144, _BF16, (8, 4, 1, 1)),
        (17, 256, 7168, _BF16, (8, 4, 1, 1)),
        (33, 256, 7168, _BF16, (4, 4, 1, 1)),
        (48, 256, 8192, _BF16, (4, 4, 1, 1)),
        (49, 256, 7168, _BF16, (8, 4, 1, 1)),
        (96, 256, 7168, _BF16, (8, 4, 1, 1)),
        # FP32 activations at any N: [8, 2] below K 6144, [4, 4] from there.
        (96, 256, 5632, torch.float32, (8, 2, 1, 1)),
        (32, 128, 4096, torch.float32, (8, 2, 1, 1)),
        (17, 256, 6144, torch.float32, (4, 4, 1, 1)),
        (96, 1024, 7168, torch.float32, (4, 4, 1, 1)),
    ],
)
def test_launch_tile_follows_the_timed_bands(m, n, k, x_dtype, tile):
    # The tile changes the time, not the bits (forced tiles on the GPU below).
    assert gluon_gemv._simt_tile(m, n, k, x_dtype) == tile


def test_bf16_activations_widen_for_an_fp32_weight():
    # CPU calls take the Torch leaf; ``out`` must have the promoted dtype.
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(20, 512, generator=generator).bfloat16()
    weight = torch.randn(64, 512, generator=generator)
    expected = torch.mm(x.float(), weight.t())
    assert torch.equal(decode_gemv(x, weight), expected)
    out = torch.empty(20, 64)
    assert decode_gemv(x, weight, out) is out
    assert torch.equal(out, expected)
    with pytest.raises(ValueError):
        decode_gemv(x, weight, torch.empty(20, 64, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "x_dtype,weight_dtype", [(torch.bfloat16, torch.bfloat16), _FP32, _MIXED]
)
@pytest.mark.parametrize("strided", ["x", "weight"])
def test_noncontiguous_inputs_skip_the_registry(
    monkeypatch, x_dtype, weight_dtype, strided
):
    def registry_lookup(*args):
        raise AssertionError("a noncontiguous input reached the registry")

    monkeypatch.setattr(triton_gemv, "_select", registry_lookup)
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(24, 2 * 512, generator=generator).to(x_dtype)
    weight = torch.randn(2 * 64, 512, generator=generator).to(weight_dtype)
    x = x[:, :512] if strided == "x" else x[:, :512].contiguous()
    weight = weight[::2] if strided == "weight" else weight[:64]
    torch.testing.assert_close(
        decode_gemv(x, weight),
        torch_decode_gemv(x.contiguous(), weight.contiguous()),
    )


@_requires_nvidia
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (17, 256, 7168),
        (25, 128, 4096),
        (33, 128, 4096),
        (49, 128, 4096),
        (96, 128, 4096),
        (40, 256, 7168),
        (64, 64, 512),
        (21, 4, 8192),
    ],
)
def test_kernel_matches_the_reference_bit_for_bit(x_dtype, m, n, k):
    # Called directly, the kernel also takes FP32 activations.
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda").to(x_dtype)
    weight = torch.randn(n, k, device="cuda") * 0.02
    out = torch.full((m, n), float("nan"), device="cuda")
    assert gluon_simt_gemm_fp32(x, weight, out) is out
    assert torch.equal(_bits(out), _bits(_order_reference(x, weight)))
    exact = x.double().cpu() @ weight.double().cpu().t()
    assert ((out.double().cpu() - exact).abs() <= _error_bound(x, weight)).all()


@_requires_nvidia
def test_rows_do_not_depend_on_batch_or_tile(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(96, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 7168, device="cuda") * 0.02
    full = _bits(gluon_simt_gemm_fp32(x, weight))
    for m in (17, 24, 33, 49, 80):
        assert torch.equal(
            _bits(gluon_simt_gemm_fp32(x[:m].contiguous(), weight)), full[:m]
        )
    for tile in ((8, 2, 1, 1), (8, 4, 1, 1), (4, 4, 1, 1), (4, 2, 1, 1), (2, 4, 1, 1)):
        monkeypatch.setattr(gluon_gemv, "_simt_tile", lambda *_, tile=tile: tile)
        assert torch.equal(_bits(gluon_simt_gemm_fp32(x, weight)), full)


@_requires_nvidia
@pytest.mark.parametrize("m", [17, 48, 96])
def test_decode_gemv_routes_rows_17_to_96(m):
    torch.manual_seed(m)
    x = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 7168, device="cuda") * 0.02
    out = torch.full((m, 256), float("nan"), device="cuda")
    _select.cache_clear()
    assert decode_gemv(x, weight, out) is out
    assert torch.equal(_bits(out), _bits(gluon_simt_gemm_fp32(x, weight)))
    # BF16 to FP32 is exact and the order does not depend on the tile, so the
    # mixed call is the kernel's FP32 call on x.float().
    assert torch.equal(_bits(out), _bits(gluon_simt_gemm_fp32(x.float(), weight)))
    with pytest.raises(ValueError):
        decode_gemv(x, weight, torch.empty(m, 256, device="cuda", dtype=x.dtype))


@_requires_split_kernel
@pytest.mark.parametrize("m", [17, 96, 97, 128])
def test_split_weight_takes_this_kernel_up_to_96_rows(m):
    torch.manual_seed(m)
    x = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 7168, device="cuda") * 0.02
    pieces = decode_gemv_weight_split(weight)
    assert pieces is not None
    out = torch.full((m, 256), float("nan"), device="cuda")
    _select.cache_clear()
    assert decode_gemv(x, weight, out, weight_split=pieces) is out
    if m <= 96:
        expected = gluon_simt_gemm_fp32(x, weight)
    else:
        expected = triton_bf16x3_gemm_fp32(x, pieces)
    assert torch.equal(_bits(out), _bits(expected))


@_requires_nvidia
def test_one_hot_rows_are_exact():
    weight = torch.randn(128, 4096, device="cuda")
    columns = torch.arange(0, 4096, 64, device="cuda")[:64]
    x = torch.zeros(64, 4096, device="cuda", dtype=torch.bfloat16)
    x[torch.arange(64), columns] = 1
    assert torch.equal(
        _bits(gluon_simt_gemm_fp32(x, weight)), _bits(weight[:, columns].t())
    )


@_requires_nvidia
def test_graph_replay_matches_eager():
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 4096, device="cuda") * 0.02
    out = torch.empty(64, 128, device="cuda")
    decode_gemv(x, weight, out)  # compile before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_gemv(x, weight, out)
    for _ in range(2):
        x.normal_()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(gluon_simt_gemm_fp32(x, weight)))


@_requires_nvidia
def test_reduced_matmul_precision_does_not_change_the_bits():
    x = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 4096, device="cuda") * 0.02
    expected = _bits(decode_gemv(x, weight))
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        assert torch.equal(_bits(decode_gemv(x, weight)), expected)
    finally:
        torch.set_float32_matmul_precision(previous)


@_requires_nvidia
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", ["x", "weight"])
def test_noncontiguous_rows_take_torch(monkeypatch, x_dtype, strided):
    torch.manual_seed(3)
    x = torch.randn(32, 2 * 4096, device="cuda").to(x_dtype)
    weight = torch.randn(2 * 128, 4096, device="cuda")
    x = x[:, :4096] if strided == "x" else x[:, :4096].contiguous()
    weight = weight[::2] if strided == "weight" else weight[:128]
    calls = []

    def torch_leaf(*args):
        calls.append(args)
        return torch_decode_gemv(*args)

    monkeypatch.setattr(triton_gemv, "torch_decode_gemv", torch_leaf)
    got = decode_gemv(x, weight)
    assert len(calls) == 1
    # Torch picks its own summation order: hold it to the bound of any order.
    exact = x.double().cpu() @ weight.double().cpu().t()
    bound = _error_bound(x, weight, rounds=x.shape[1])
    assert ((got.double().cpu() - exact).abs() <= bound).all()


@_requires_nvidia
@pytest.mark.parametrize(
    "x_dtype,m,n,k",
    [
        (torch.float32, 17, 64, 8192),
        (torch.float32, 48, 256, 7168),
        (torch.float32, 96, 128, 4096),
        (torch.bfloat16, 96, 512, 3072),
        (torch.bfloat16, 96, 1024, 2048),
    ],
)
def test_inputs_left_to_torch_take_torch(x_dtype, m, n, k):
    # Inside the rows and K of the traits, but slower than Torch on a GB200.
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda").to(x_dtype)
    weight = torch.randn(n, k, device="cuda") * 0.02
    _select.cache_clear()
    assert _select(m, n, k, True, x_dtype, weight.dtype, False) is torch_decode_gemv
    out = torch.full((m, n), float("nan"), device="cuda")
    assert decode_gemv(x, weight, out) is out
    expected = torch_decode_gemv(x, weight, torch.empty_like(out))
    assert torch.equal(_bits(out), _bits(expected))
