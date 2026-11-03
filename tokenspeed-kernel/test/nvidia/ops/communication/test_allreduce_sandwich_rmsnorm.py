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

"""The pre/post ("sandwich") norm boundary fused into the MNNVL all-reduce.

CPU only, nothing is launched:

* NumPy models of the epilogue (trtllm_mnnvl_sandwich_norm.cuh, operation by
  operation) and of the unfused chain (the same operations in the Triton
  rmsnorm reduction order) against FP64 references: the fused outputs stay
  within the bounds an order-only change allows.
* The arithmetic the epilogue pins (no flush-to-zero, the mean folded into
  fma(s, RN(1 / hidden), eps), rsqrt.approx.ftz) is the arithmetic ptxas
  emits for the Triton rmsnorm the unfused chain runs.
* The pattern code, the launch parameters and the wrapper's admission: what
  it serves, what it hands back to the caller (None), what it refuses.
* The per-token FP8 stage (kARSandwichResidualRMSNormTokenFP8): a NumPy model
  of its warp / cluster amax exchange and its FP32 operations gives the codes
  and scales of the per-token rule ``fp8_quantize_per_token`` applies to the
  BF16 output, at every cluster partition; the wrapper's FP8 arguments.

The multi-GPU checks are ``nvidia/thirdparty/test_trtllm_mnnvl_sandwich_norm.py``.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenspeed_kernel.numerics.torch.communication import torch_sandwich_norm
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="the MNNVL all-reduce is NVIDIA only"
)

F32 = np.float32
KERNEL_ROOT = Path(__file__).resolve().parents[4]  # tokenspeed-kernel/
CSRC = KERNEL_ROOT / "python/tokenspeed_kernel/thirdparty/cuda/csrc"
HEADER = CSRC / "include/flashinfer/comm/trtllm_mnnvl_sandwich_norm.cuh"
FUSION_HEADER = CSRC / "include/flashinfer/comm/trtllm_allreduce_fusion.cuh"
BINDING = CSRC / "trtllm_mnnvl_sandwich_norm.cu"


# ------------------------------------------------------------ NumPy models


def _bf16(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=F32))
    return t.to(torch.bfloat16).to(torch.float32).numpy()


def _fma(a, b, c) -> np.ndarray:
    """Correctly rounded float32 fma: exact product, round-to-odd sum in
    float64 (53 >= 24 + 2), then one rounding to float32."""
    a, b, c = (np.asarray(v, F32).astype(np.float64) for v in (a, b, c))
    p = a * b
    s = p + c
    bp = s - c
    err = (p - bp) + (c - (s - bp))
    fix = (err != 0) & ((s.view(np.int64) & 1) == 0) & np.isfinite(s)
    s = np.where(
        fix, np.where(err > 0, np.nextafter(s, np.inf), np.nextafter(s, -np.inf)), s
    )
    return s.astype(F32)


def _rsqrt(v: np.ndarray, ulp: int) -> np.ndarray:
    """rsqrt.approx: RN(1 / sqrt(v)) moved by ``ulp`` units (its error bound)."""
    r = (1.0 / np.sqrt(v.astype(np.float64))).astype(F32)
    return (r.view(np.int32) + np.int32(ulp)).view(F32)


def _group_terms(x: np.ndarray):
    sq = lambda i: (x[..., i] * x[..., i]).astype(F32)  # noqa: E731
    return (
        _fma(x[..., 0], x[..., 0], sq(2)),
        _fma(x[..., 1], x[..., 1], sq(3)),
        _fma(x[..., 4], x[..., 4], sq(6)),
        _fma(x[..., 5], x[..., 5], sq(7)),
    )


def _butterfly(v: np.ndarray) -> np.ndarray:
    lanes = np.arange(32)
    for m in (16, 8, 4, 2, 1):
        v = (v + v[..., lanes ^ m]).astype(F32)
    return v[..., 0]


def _fused_sumsq(x: np.ndarray) -> np.ndarray:
    """The epilogue's order: 8-element groups, warp butterflies over 32
    groups, then the hidden / 256 warp sums lane-strided and a butterfly."""
    rows, hidden = x.shape
    ax, ay, bx, by = _group_terms(x.reshape(rows, hidden // 8, 8))
    q = ((ax + bx).astype(F32) + (ay + by).astype(F32)).astype(F32)
    warp = _butterfly(q.reshape(rows, hidden // 256, 32))
    lane = np.zeros((rows, 32), F32)
    for w in range(hidden // 256):
        lane[:, w % 32] = (lane[:, w % 32] + warp[:, w]).astype(F32)
    return _butterfly(lane)


def _triton_sumsq(x: np.ndarray) -> np.ndarray:
    """The Triton rmsnorm order at 4 warps: thread t holds elements
    r * 1024 + 8t + j; per rep the two pair halves, reps pairwise, the
    halves, a warp butterfly, then (w0 + w2) + (w1 + w3)."""
    rows, hidden = x.shape
    block = 1 << (hidden - 1).bit_length()
    padded = np.zeros((rows, block), F32)
    padded[:, :hidden] = x
    ax, ay, bx, by = _group_terms(padded.reshape(rows, block // 1024, 128, 8))
    halves = []
    for v in ((ax + bx).astype(F32), (ay + by).astype(F32)):
        while v.shape[1] > 1:
            v = (v[:, 0::2] + v[:, 1::2]).astype(F32)
        halves.append(v[:, 0])
    w = _butterfly((halves[0] + halves[1]).astype(F32).reshape(rows, 4, 32))
    return ((w[:, 0] + w[:, 2]).astype(F32) + (w[:, 1] + w[:, 3]).astype(F32)).astype(
        F32
    )


def _boundary(s, r, gp, g, eps_p, eps, xs, rs, sumsq, ulp, drop=None):
    """a = RMSNorm(s; gp), residual_out = RN(RN(xs a) + RN(rs r)),
    norm_out = RMSNorm(residual_out; g), in FP32 with the BF16 roundings
    (``drop="a"`` or ``"xp"`` skips the rounding of ``a`` or of ``xs a``)."""
    inv_h = F32(1.0) / F32(s.shape[1])

    def norm(x, gamma, e, rounded=True):
        rstd = _rsqrt(_fma(sumsq(x), inv_h, F32(e)), ulp)
        y = ((rstd[:, None] * x).astype(F32) * gamma[None, :]).astype(F32)
        return _bf16(y) if rounded else y

    a = norm(s, gp, eps_p, rounded=drop != "a")
    xp = (a * F32(xs)).astype(F32)
    xp = xp if drop == "xp" else _bf16(xp)
    rp = _bf16((r * F32(rs)).astype(F32))
    res = _bf16((xp + rp).astype(F32))
    return norm(res, g, eps), res, xp, rp, a


def _staged_reference(s, r, gp, g, eps_p, eps, xs, rs):
    """FP64 arithmetic with the BF16 rounding points of the boundary."""
    return tuple(
        t.float().numpy() for t in _sandwich(s, r, gp, g, eps_p, eps, xs, rs, True)
    )


def _exact_reference(s, r, gp, g, eps_p, eps, xs, rs):
    """FP64 throughout, no intermediate rounding."""
    out, res, _, _ = _sandwich(s, r, gp, g, eps_p, eps, xs, rs, False)
    return out.numpy(), res.numpy()


def _sandwich(s, r, gp, g, eps_p, eps, xs, rs, round_bf16):
    return torch_sandwich_norm(
        *(torch.from_numpy(np.ascontiguousarray(v)) for v in (s, r, gp, g)),
        post_eps=eps_p,
        eps=eps,
        x_scale=xs,
        residual_scale=rs,
        round_bf16=round_bf16,
    )


def _ulp(v: np.ndarray) -> np.ndarray:
    """BF16 ulp of |v| (8 significant bits)."""
    v = np.abs(v.astype(np.float64))
    e = np.floor(np.log2(np.where(v > 0, v, 1.0)))
    return np.where(v >= 2.0**-126, np.exp2(e - 7), 2.0**-133)


def _inputs(hidden: int, kind: str, rows: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed + hidden)
    s = rng.standard_normal((rows, hidden))
    if kind == "large":
        s *= 64
    elif kind == "heavy":
        s = rng.standard_t(2, (rows, hidden))
    elif kind == "outlier":
        s[:, rng.choice(hidden, 8, replace=False)] *= 200
    r = rng.standard_normal((rows, hidden)) * 4
    gp = 1 + 0.1 * rng.standard_normal(hidden)
    g = np.abs(1 + 0.3 * rng.standard_normal(hidden))
    return tuple(_bf16(v.astype(F32)) for v in (s, r, gp, g))


# Module bounds for an order-only change of the two sums of squares. The GPU
# test holds every call to the element bounds and the calls of one hidden size
# and input kind together to the mismatch and RMS limits.
MAX_MISMATCH = 1e-3  # fraction of elements of one output
RESIDUAL_ADDEND_ULPS = 2  # |d residual_out| in ulps of the larger addend or sum
XP_MARGIN = 2.0**-5  # the other path's xp can sit one binade above the staged one
NORM_ULPS = 1  # |d norm_out| beyond the propagated residual difference
RMS_RATIO = (0.99, 1.01)  # RMS error vs FP64, fused / unfused

# The scale pairs of the GPU test. An x_scale that is not a power of two makes
# RN(x_scale * a) differ from x_scale * RN(a), so a dropped rounding of a or of
# x_scale * a shows; with a power of two both give the same bits.
SCALE_PAIRS = ((0.3, 0.97), (1.1, 1.0))


def _check_pair(out, res, other_out, other_res, xp, rp, g):
    """``out, res`` within the bounds of ``other_out, other_res``."""
    assert (res != other_res).mean() <= MAX_MISMATCH
    assert (out != other_out).mean() <= MAX_MISMATCH
    d_res = np.abs(res.astype(np.float64) - other_res)
    # A one-ulp flip of a moves x_scale * a by up to two ulps, possibly one
    # binade above the staged xp (the margin); a sum one binade above both
    # addends can round two exact ties to even in opposite directions.
    magnitude = np.max(
        [np.abs(res), np.abs(other_res), (1 + XP_MARGIN) * np.abs(xp), np.abs(rp)],
        axis=0,
    )
    assert (d_res <= RESIDUAL_ADDEND_ULPS * _ulp(magnitude)).all()
    x64 = other_res.astype(np.float64)
    rstd = 1.0 / np.sqrt((x64 * x64).mean(axis=1))
    propagated = 1.01 * d_res * rstd[:, None] * np.abs(g.astype(np.float64))
    d_out = np.abs(out.astype(np.float64) - other_out)
    assert (
        d_out
        <= NORM_ULPS * _ulp(np.maximum(np.abs(out), np.abs(other_out))) + propagated
    ).all()


@pytest.mark.parametrize("hidden", [1024, 4096, 7168])
@pytest.mark.parametrize("kind", ["normal", "large", "heavy", "outlier"])
@pytest.mark.parametrize("pair", [(1.0, 1.0), *SCALE_PAIRS])
@pytest.mark.parametrize("ulp", [0, -2, 2])
def test_fused_order_stays_within_the_order_only_bounds(hidden, kind, pair, ulp):
    s, r, gp, g = _inputs(hidden, kind)
    args = (s, r, gp, g, 1e-6, 1e-5, *pair)
    f_out, f_res, *_ = _boundary(*args, sumsq=_fused_sumsq, ulp=ulp)
    u_out, u_res, *_ = _boundary(*args, sumsq=_triton_sumsq, ulp=ulp)
    ref_out, ref_res, ref_xp, ref_rp = _staged_reference(*args)
    # Against the unfused chain, and both against the staged FP64 reference.
    _check_pair(f_out, f_res, u_out, u_res, ref_xp, ref_rp, g)
    _check_pair(f_out, f_res, ref_out, ref_res, ref_xp, ref_rp, g)
    _check_pair(u_out, u_res, ref_out, ref_res, ref_xp, ref_rp, g)
    # The same accuracy against FP64 without intermediate rounding.
    exact_out, exact_res = _exact_reference(*args)
    for fused, unfused, exact in ((f_res, u_res, exact_res), (f_out, u_out, exact_out)):
        ratio = np.sqrt(((fused - exact) ** 2).mean() / ((unfused - exact) ** 2).mean())
        assert RMS_RATIO[0] <= ratio <= RMS_RATIO[1]


@pytest.mark.parametrize("pair", SCALE_PAIRS)
@pytest.mark.parametrize("drop", ["a", "xp"])
def test_the_scale_pairs_expose_a_dropped_rounding_point(pair, drop):
    """An epilogue that skipped the BF16 rounding of ``a`` or of
    ``x_scale * a`` falls outside the bounds at the GPU test's pairs."""
    s, r, gp, g = _inputs(4096, "normal")
    args = (s, r, gp, g, 1e-6, 1e-5, *pair)
    f_out, f_res, *_ = _boundary(*args, sumsq=_fused_sumsq, ulp=0, drop=drop)
    u_out, u_res, *_ = _boundary(*args, sumsq=_triton_sumsq, ulp=0)
    _, _, ref_xp, ref_rp = _staged_reference(*args)
    with pytest.raises(AssertionError):
        _check_pair(f_out, f_res, u_out, u_res, ref_xp, ref_rp, g)


def test_without_a_scale_pair_the_add_is_the_plain_bf16_sum():
    """(1.0, 1.0) is the kernel's missing pair: both products are exact."""
    s, r, gp, g = _inputs(4096, "normal")
    out, res, xp, rp, a = _boundary(s, r, gp, g, 1e-6, 1e-6, 1.0, 1.0, _fused_sumsq, 0)
    assert np.array_equal(xp, a)
    assert np.array_equal(rp, r)
    assert np.array_equal(res, _bf16((a + r).astype(F32)))


def test_the_fused_order_does_not_depend_on_the_rows_around_a_row():
    s, r, gp, g = _inputs(2048, "heavy", rows=8)
    whole = _boundary(s, r, gp, g, 1e-6, 1e-6, 0.5, 1.0, _fused_sumsq, 0)
    for row in range(8):
        alone = _boundary(
            s[row : row + 1],
            r[row : row + 1],
            gp,
            g,
            1e-6,
            1e-6,
            0.5,
            1.0,
            _fused_sumsq,
            0,
        )
        for full, one in zip(whole, alone):
            assert np.array_equal(full[row : row + 1], one)


# ----------------------------------------------- the arithmetic that is pinned


def test_epilogue_arithmetic_is_inline_ptx_without_flush_to_zero():
    """The kernel package builds with -use_fast_math; the epilogue's FP32
    operations are inline PTX, and only rsqrt flushes subnormals (the per-token
    FP8 stage's comparisons, division and E4M3 conversion included)."""
    text = HEADER.read_text()
    ops = set(
        re.findall(
            r'asm\("((?:mul|add|fma|rcp|rsqrt|cvt|div|abs|max|min)\.[a-z0-9.]+)', text
        )
    )
    assert ops == {
        "mul.rn.f32",
        "add.rn.f32",
        "fma.rn.f32",
        "rcp.rn.f32",
        "rsqrt.approx.ftz.f32",
        "cvt.rn.bf16x2.f32",
        "div.rn.f32",
        "abs.f32",
        "max.f32",
        "min.f32",
        "cvt.rn.satfinite.e4m3x2.f32",
    }
    for call in ("fmaxf(", "fminf(", "fabsf(", "__fdiv"):
        assert call not in text
    assert "membar" not in text and "__threadfence" not in text
    assert "rcp_rn(static_cast<float>(m_params.hidden_dim))" in text


def _compile_for_sm100(fn, signature, constexprs=None, attrs=None, **options):
    from tokenspeed_kernel._triton import triton

    backends = pytest.importorskip("tokenspeed_triton.backends.compiler")
    compiler = pytest.importorskip("tokenspeed_triton.compiler")
    source = compiler.ASTSource(
        fn=fn, signature=signature, constexprs=constexprs, attrs=attrs
    )
    return triton.compile(
        source,
        target=backends.GPUTarget("cuda", 100, 32),
        options={"num_warps": 4, "launch_pdl": True, **options},
    )


def _skip_unless_triton_compiles_for_sm100():
    """Skip where Triton cannot build sm_100 code at all: a kernel without
    parameters does not compile. Only this probe skips, so a signature that
    no longer matches the rmsnorm kernel fails the test."""
    from tokenspeed_kernel._triton import triton

    @triton.jit
    def empty():
        pass

    try:
        # sm_100a needs PTX 8.6. Asking for it makes a ptxas older than CUDA
        # 12.8 fail here; at the PTX version Triton derives from such a
        # ptxas, LLVM would abort the test process instead.
        _compile_for_sm100(empty, {}, ptx_version=86)
    except Exception as exc:  # noqa: BLE001 - this host's Triton toolchain
        pytest.skip(f"Triton cannot compile for sm_100 here: {exc}")


def _rmsnorm_launch(hidden: int, residual: bool):
    """The ``_rmsnorm_kernel`` arguments (by name) and launch options of the
    unfused chain's ``rmsnorm()`` call on one row: the post norm (no
    residual) or the next pre-norm (the scaled BF16 residual add). The launch
    needs a GPU, so it is recorded instead of run."""
    from tokenspeed_kernel.ops.layernorm import triton as layernorm

    launches = []

    class RecordLaunch:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((args, kwargs))

    x = torch.empty(1, hidden, dtype=torch.bfloat16, device="meta")
    weight = torch.empty(hidden, dtype=torch.bfloat16, device="meta")
    add = {}
    if residual:
        add = dict(
            residual=torch.empty_like(x),
            round_residual_sum_bf16=True,
            x_scale=1.0,
            residual_scale=1.0,
        )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(layernorm, "_rmsnorm_kernel", RecordLaunch())
        layernorm.rmsnorm(x, weight, 1e-6, enable_pdl=True, **add)
    ((args, kwargs),) = launches
    names = layernorm._rmsnorm_kernel.arg_names
    values = dict(zip(names, args))
    values.update((k, v) for k, v in kwargs.items() if k in names)
    return values, {k: v for k, v in kwargs.items() if k not in names}


def _compile_triton_rmsnorm(hidden: int, residual: bool):
    from tokenspeed_kernel.ops.layernorm.triton import _rmsnorm_kernel

    # The kernel as rmsnorm() launches it. Constexprs and None arguments are
    # constants; the other arguments take the types Triton gives them at the
    # launch. A parameter rmsnorm() leaves out takes its default.
    values, options = _rmsnorm_launch(hidden, residual)
    signature, constexprs, attrs = {}, {}, {}
    for index, param in enumerate(_rmsnorm_kernel.params):
        assert param.name in values or param.has_default, param.name
        value = values.get(param.name, param.default)
        if param.is_constexpr or value is None:
            signature[param.name] = "constexpr"
            constexprs[param.name] = value
            continue
        if isinstance(value, torch.Tensor):
            signature[param.name] = {torch.bfloat16: "*bf16"}[value.dtype]
        else:
            signature[param.name] = "fp32" if isinstance(value, float) else "i32"
        # The product launch's tensors are 16-byte aligned and its row stride
        # is the hidden size, a multiple of 16.
        if signature[param.name] != "fp32":
            attrs[(index,)] = [["tt.divisibility", 16]]
    return _compile_for_sm100(_rmsnorm_kernel, signature, constexprs, attrs, **options)


@pytest.mark.parametrize("hidden", [1024, 4096, 7168])
@pytest.mark.parametrize("residual", [False, True])
def test_triton_rmsnorm_compiles_to_the_pinned_arithmetic(tmp_path, hidden, residual):
    """ptxas folds the Triton kernel's mean and epsilon into
    FFMA(s, RN(1 / hidden), eps) before MUFU.RSQ, and no FP32 operation but
    rsqrt flushes subnormals: the operations the epilogue reproduces."""
    if shutil.which("cuobjdump") is None:
        pytest.skip("cuobjdump is not on PATH")
    _skip_unless_triton_compiles_for_sm100()
    kernel = _compile_triton_rmsnorm(hidden, residual)
    ptx = kernel.asm["ptx"]
    ftz = set(re.findall(r"\b[a-z0-9]+\.[a-z0-9.]*ftz[a-z0-9.]*", ptx))
    assert ftz == {"rsqrt.approx.ftz.f32"}
    cubin = tmp_path / "k.cubin"
    cubin.write_bytes(kernel.asm["cubin"])
    sass = subprocess.run(
        ["cuobjdump", "-sass", str(cubin)], capture_output=True, text=True, check=True
    ).stdout
    lines = [ln for ln in sass.splitlines() if re.search(r"/\*[0-9a-f]{4}\*/", ln)]
    rsq = next(i for i, ln in enumerate(lines) if "MUFU.RSQ" in ln)
    src = re.search(r"MUFU\.RSQ\s+R\d+,\s*(R\d+)", lines[rsq]).group(1)
    feed = next(
        ln for ln in reversed(lines[:rsq]) if re.search(rf"\bFFMA\s+{src},", ln)
    )
    a, b, addend = re.search(
        r"FFMA\s+R\d+,\s*(R\d+),\s*(R\d+),\s*([-0-9.e]+)\s*;", feed
    ).groups()
    assert F32(float(addend)) == F32(1e-6)
    want = int(np.array([F32(1.0) / F32(hidden)]).view(np.uint32)[0])
    consts = set()
    for line in lines[:rsq]:
        if re.search(rf"\b(?:HFMA2|MOV|IMAD\.MOV\.U32)\s+(?:{a}|{b}),", line):
            enc = re.search(r"/\* (0x[0-9a-f]{16}) \*/", line)
            lit = re.search(r",\s*(0x[0-9a-f]+)\s*;", line)
            consts.add(
                int(enc.group(1)[2:10], 16)
                if "HFMA2" in line
                else int(lit.group(1), 16)
            )
    assert want in consts


def test_two_shot_kernels_fit_two_ctas_of_21_warps_per_sm():
    """Past one cluster per SM the launch runs in waves of resident CTAs. Up
    to 8 ranks the two-shot kernels keep to 48 registers on sm_100a and
    sm_103a, so two CTAs of up to 21 warps share an SM as the plain
    all-reduce's do; more registers halve the CTAs each wave holds. The
    per-token FP8 pattern (12) keeps to the same budget."""
    so = (
        KERNEL_ROOT
        / "python/tokenspeed_kernel/thirdparty/cuda/objs/trtllm_comm/trtllm_comm.so"
    )
    if shutil.which("cuobjdump") is None or not so.exists():
        pytest.skip("needs the built trtllm_comm.so and cuobjdump")
    usage = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(so)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    kernel = re.compile(
        r"kernel_twoshotI.*PatternE1[12]E13__nv_bfloat16Li(\d+)E.*\n\s*REG:(\d+)"
    )
    checked = 0
    for block in re.split(r"\narch = ", usage):
        if not block.startswith(("sm_100a", "sm_103a")):
            continue
        for ranks, regs in kernel.findall(block):
            if int(ranks) <= 8:
                assert int(regs) <= 48, (block[:7], ranks, regs)
                checked += 1
    if checked == 0:
        pytest.skip("the build has no sm_100a / sm_103a code")


# ------------------------------------------------- pattern code and parameters


def test_pattern_code_parameters_and_build_agree():
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        _MNNVL_SUPPORTED_PATTERNS,
        AllReduceFusionPattern,
    )

    fusion = FUSION_HEADER.read_text()
    assert re.search(r"kARSandwichResidualRMSNorm = 11,", fusion)
    assert re.search(r"kARSandwichResidualRMSNormTokenFP8 = 12,", fusion)
    assert AllReduceFusionPattern.kARSandwichResidualRMSNorm == 11
    assert AllReduceFusionPattern.kARSandwichResidualRMSNormTokenFP8 == 12
    assert (
        AllReduceFusionPattern.kARSandwichResidualRMSNorm in _MNNVL_SUPPORTED_PATTERNS
    )
    params = fusion[fusion.index("struct AllReduceFusionParams") :]
    params = params[: params.index("};")]
    tail = params[params.index("bool residual_reduce_scattered") :]
    fields = (
        "void* post_norm_gamma = nullptr;",
        "float x_scale = 1.f;",
        "float residual_scale = 1.f;",
    )
    # Appended, 16 bytes in all: the other fields keep their offsets and the
    # kernel parameters after the struct their 16-byte alignment.
    assert [line.strip() for line in tail.splitlines() if line.strip().endswith(";")][
        1:
    ] == list(fields)
    # The generic dispatchers do not instantiate the pattern.
    mnnvl = (
        CSRC / "include/flashinfer/comm/trtllm_mnnvl_allreduce_fusion.cuh"
    ).read_text()
    assert "kARSandwichResidualRMSNorm" not in mnnvl
    assert (
        "kARSandwichResidualRMSNorm"
        not in fusion[fusion.index("allreduce_fusion_op(") :]
    )
    binding = BINDING.read_text()
    assert (
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_sandwich_norm_allreduce," in binding
    )
    assert (
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_sandwich_norm_allreduce_token_fp8,"
        in binding
    )
    # The FP8 constants are the Triton rule's FP32 constexprs.
    assert "kE4m3Max = 448.f;" in HEADER.read_text()
    assert (
        "kTokenMinScale = static_cast<float>(1.0 / (448.0 * 512.0));"
        in HEADER.read_text()
    )
    assert "kTokenMaxMultiplier = 448.f * 512.f;" in HEADER.read_text()
    setup = (KERNEL_ROOT / "python/setup.py").read_text()
    assert 'CUDA_CSRC_DIR / "trtllm_mnnvl_sandwich_norm.cu"' in setup
    header = HEADER.read_text()
    assert "static constexpr int kMaxWarps = 256;" in header


# ------------------------------------------------------------ the wrapper


class _Group:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _Recorder:
    def __init__(self) -> None:
        self.calls = []

    def trtllm_mnnvl_sandwich_norm_allreduce(self, *args):
        self.calls.append(args)

    def trtllm_mnnvl_sandwich_norm_allreduce_token_fp8(self, *args):
        self.calls.append(("token_fp8",) + args)


def _workspace(world: int, hidden: int, *, cap: int = 64, buffer: int = 1 << 30):
    from tokenspeed_kernel.thirdparty.cuda.trtllm import MnnvlAllReduceFusionWorkspace

    return MnnvlAllReduceFusionWorkspace(
        tp_rank=0,
        tp_size=world,
        max_token_num=2048,
        hidden_dim=hidden,
        buffer_size_bytes=buffer,
        multicast_ptr=0x1000,
        peer_ptrs=torch.zeros(world, dtype=torch.int64),
        local_ptr=0x2000,
        buffer_flags=torch.zeros(9, dtype=torch.int32),
        oneshot_token_cap=cap,
        refs=(),
    )


@pytest.fixture
def wrapper(monkeypatch):
    import tokenspeed_kernel.ops.communication.trtllm as ops
    import tokenspeed_kernel.thirdparty.cuda.trtllm as third

    state = SimpleNamespace(
        manager=SimpleNamespace(mnnvl_workspace=None, graph_consumed=False),
        armed=True,
        ensured=[],
        captured=[],
        module=_Recorder(),
    )

    def ensure(**kwargs):
        state.ensured.append(kwargs)
        return state.armed

    monkeypatch.setattr(ops, "ensure_workspace_initialized", ensure)
    monkeypatch.setattr(ops, "_manager_for_group", lambda group: state.manager)
    monkeypatch.setattr(
        ops, "_mark_captured", lambda manager, ws: state.captured.append(ws) or ws
    )
    monkeypatch.setattr(third, "_load_trtllm_comm_module", lambda: state.module)
    state.op = ops.allreduce_sandwich_rmsnorm
    return state


def _call(state, tokens=4, hidden=4096, world=8, dtype=torch.bfloat16, **kwargs):
    x = torch.zeros(tokens, hidden, dtype=dtype)
    residual = torch.zeros(tokens, hidden, dtype=dtype)
    post = torch.ones(hidden, dtype=dtype)
    weight = torch.ones(hidden, dtype=dtype)
    args = dict(eps=1e-5, max_token_num=2048, norm_out_bf16=True)
    args.update(kwargs)
    return state.op(x, residual, post, weight, 3, _Group(world), **args)


def test_a_served_call_launches_once_with_the_resolved_strategy(wrapper):
    ws = _workspace(8, 4096, cap=6)
    wrapper.manager.mnnvl_workspace = ws
    out = _call(wrapper, tokens=4, x_scale=0.5, residual_scale=0.75)
    assert out is not None and out[0].shape == out[1].shape == (4, 4096)
    assert wrapper.ensured == [
        dict(
            rank=3,
            group=wrapper.ensured[0]["group"],
            max_token_num=2048,
            hidden_dim=4096,
            use_fp32_lamport=False,
        )
    ]
    assert wrapper.captured == [ws]
    (args,) = wrapper.module.calls
    # world, rank, tokens, hidden; the workspace pointers; then (after the
    # PDL flag) the strategy, trigger at end, the epsilon and the pair.
    assert args[6:10] == (8, 3, 4, 4096)
    assert args[10:12] == (0x1000, 0x2000)
    assert args[15:] == (True, True, 1e-5, 0.5, 0.75)
    # Two-shot past the one-shot cap.
    _call(wrapper, tokens=7)
    assert wrapper.module.calls[-1][15] is False


def test_a_missing_pair_launches_with_unit_multipliers(wrapper):
    wrapper.manager.mnnvl_workspace = _workspace(4, 2048)
    _call(wrapper, world=4, hidden=2048)
    assert wrapper.module.calls[-1][-2:] == (1.0, 1.0)


@pytest.mark.parametrize(
    "case",
    [
        dict(world=1),
        dict(tokens=0),
        dict(tokens=33, max_token_num=32),
        dict(no_mnnvl=True),
        dict(unarmed=True),
        dict(hidden=2880),  # no whole-warp cluster partition
        dict(hidden=4096, buffer=1024),  # the workspace cannot hold the call
        dict(world=4),  # the workspace was armed for another world size
    ],
)
def test_calls_no_kernel_serves_return_none_without_a_launch(wrapper, case):
    case = dict(case)
    hidden = case.get("hidden", 4096)
    if not case.pop("no_mnnvl", False):
        wrapper.manager.mnnvl_workspace = _workspace(
            8, hidden, buffer=case.pop("buffer", 1 << 30)
        )
    wrapper.armed = not case.pop("unarmed", False)
    assert _call(wrapper, **case) is None
    assert wrapper.module.calls == []
    assert wrapper.captured == []


@pytest.mark.parametrize(
    "scales,launched",
    [
        (dict(x_scale=0.5), (0.5, 1.0)),
        (dict(residual_scale=0.5), (1.0, 0.5)),
        (dict(x_scale=0.5, residual_scale=None), (0.5, 1.0)),
    ],
    ids=["x-scale-only", "residual-scale-only", "residual-scale-none"],
)
@pytest.mark.parametrize("tokens", [1, 2048, 2049])
def test_a_one_sided_pair_takes_one_for_the_missing_side(
    wrapper, scales, launched, tokens
):
    """As the unfused rmsnorm(): the missing multiplier is 1.0, at T = 1, at
    the fused limit (max_token_num 2048), and above it, where the call
    returns None for the unfused chain instead of raising."""
    wrapper.manager.mnnvl_workspace = _workspace(8, 4096)
    out = _call(wrapper, tokens=tokens, **scales)
    if tokens > 2048:
        assert out is None and wrapper.module.calls == []
        return
    assert out is not None
    assert wrapper.module.calls[-1][-2:] == launched


@pytest.mark.parametrize(
    "case",
    [
        dict(dtype=torch.float16),
        dict(x_scale=math.inf),
        dict(residual_scale=math.nan),
        dict(x_scale=math.inf, residual_scale=1.0),
        dict(x_scale=True, residual_scale=1.0),
        dict(x_scale=1, residual_scale=1.0),
    ],
)
def test_invalid_calls_raise_before_any_collective(wrapper, case):
    wrapper.manager.mnnvl_workspace = _workspace(8, 4096)
    with pytest.raises(ValueError):
        _call(wrapper, **case)
    assert wrapper.ensured == [] and wrapper.module.calls == []


def test_mismatched_shapes_raise(wrapper):
    wrapper.manager.mnnvl_workspace = _workspace(8, 4096)
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    good = torch.ones(4096, dtype=torch.bfloat16)
    for residual, post in (
        (torch.zeros(3, 4096, dtype=torch.bfloat16), good),
        (
            torch.zeros(4, 4096, dtype=torch.bfloat16),
            torch.ones(2048, dtype=torch.bfloat16),
        ),
        (torch.zeros(4096, 4, dtype=torch.bfloat16).t(), good),
    ):
        with pytest.raises(ValueError):
            wrapper.op(
                x, residual, post, good, 0, _Group(8), eps=1e-6, norm_out_bf16=True
            )
    assert wrapper.module.calls == []


def _fp8_buffers(tokens=4, hidden=4096):
    return (
        torch.empty(tokens, hidden, dtype=torch.float8_e4m3fn),
        torch.empty(tokens, 1, dtype=torch.float32),
    )


@pytest.mark.parametrize("bf16_out", [True, False])
def test_a_token_fp8_call_launches_the_fp8_entry(wrapper, bf16_out):
    """token_fp8_out launches the per-token FP8 entry with the BF16 call's
    arguments plus the codes and scales after residual_out; without the BF16
    output its norm_out is None and so is the returned one."""
    wrapper.manager.mnnvl_workspace = _workspace(8, 4096, cap=6)
    codes, scales = _fp8_buffers()
    out = _call(
        wrapper,
        x_scale=0.5,
        residual_scale=0.75,
        token_fp8_out=(codes, scales),
        norm_out_bf16=bf16_out,
    )
    (args,) = wrapper.module.calls
    assert args[0] == "token_fp8"
    args = args[1:]
    assert (args[4] is None) == (not bf16_out) and args[4] is out[0]
    assert args[5] is out[1] and args[6] is codes and args[7] is scales
    assert args[8:12] == (8, 3, 4, 4096)
    assert args[12:14] == (0x1000, 0x2000)
    assert args[17:] == (True, True, 1e-5, 0.5, 0.75)
    # The BF16 call is unchanged: the BF16 entry, no FP8 arguments.
    _call(wrapper, x_scale=0.5, residual_scale=0.75)
    assert wrapper.module.calls[-1][0] != "token_fp8"
    assert len(wrapper.module.calls[-1]) == 20


@pytest.mark.parametrize(
    "case",
    [
        dict(codes=torch.empty(4, 4096, dtype=torch.bfloat16)),
        dict(codes=torch.empty(3, 4096, dtype=torch.float8_e4m3fn)),
        dict(codes=torch.empty(4096, 4, dtype=torch.float8_e4m3fn).t()),
        dict(scales=torch.empty(4, 1, dtype=torch.bfloat16)),
        dict(scales=torch.empty(5, 1, dtype=torch.float32)),
        dict(scales=torch.empty(4, 2, dtype=torch.float32)[:, :1]),
        dict(no_fp8=True),
    ],
)
def test_invalid_token_fp8_calls_raise_before_any_collective(wrapper, case):
    wrapper.manager.mnnvl_workspace = _workspace(8, 4096)
    codes, scales = _fp8_buffers()
    codes, scales = case.get("codes", codes), case.get("scales", scales)
    kwargs = (
        dict(norm_out_bf16=False)
        if case.get("no_fp8")
        else dict(token_fp8_out=(codes, scales))
    )
    with pytest.raises(ValueError):
        _call(wrapper, **kwargs)
    assert wrapper.ensured == [] and wrapper.module.calls == []


# --------------------------------------------------- the per-token FP8 stage


def _e4m3(x: np.ndarray) -> np.ndarray:
    """cvt.rn.satfinite.e4m3x2.f32 of already clamped FP32 values: RN to E4M3
    (PyTorch's conversion), as raw bytes."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=F32))
    return t.to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def _fused_token_fp8(h: np.ndarray, cluster: int):
    """The epilogue's per-token FP8 stage on BF16 rows ``h`` (as FP32), for a
    cluster of ``cluster`` CTAs: each thread's max |x| over its 8 elements, a
    warp butterfly, the warp values exchanged and folded lane-strided, then a
    butterfly; div.rn, min / max and the clamp in FP32 (no flush), the E4M3
    conversion of each element."""
    rows, hidden = h.shape
    warps = hidden // 256
    assert warps % cluster == 0
    absx = np.abs(h).astype(F32)
    thread = absx.reshape(rows, hidden // 8, 8).max(-1)
    warp = thread.reshape(rows, warps, 32)
    lanes = np.arange(32)
    for m in (16, 8, 4, 2, 1):
        warp = np.maximum(warp, warp[..., lanes ^ m])
    slots = warp[..., 0]
    lane = np.zeros((rows, 32), F32)
    for w in range(warps):
        lane[:, w % 32] = np.maximum(lane[:, w % 32], slots[:, w])
    for m in (16, 8, 4, 2, 1):
        lane = np.maximum(lane, lane[..., lanes ^ m])
    amax = lane[:, :1]
    with np.errstate(divide="ignore", over="ignore"):
        multiplier = np.minimum((F32(448.0) / amax).astype(F32), F32(448.0 * 512.0))
    scale = np.maximum((amax / F32(448.0)).astype(F32), F32(1.0 / (448.0 * 512.0)))
    q = np.clip((h * multiplier).astype(F32), F32(-448.0), F32(448.0))
    return _e4m3(q), scale.astype(F32)


def _triton_rule(h: np.ndarray):
    """ops/quantization/triton.py's per-token rule on the same BF16 rows, in
    PyTorch FP32: fp8_token_multiplier, fp8_quantize_by_multiplier,
    fp8_token_scale."""
    x = torch.from_numpy(h)
    amax = x.abs().amax(-1, keepdim=True)
    multiplier = torch.minimum(
        torch.tensor(448.0) / amax, torch.tensor(448.0 * 512.0, dtype=torch.float32)
    )
    codes = (x * multiplier).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    scale = torch.maximum(
        amax / 448.0, torch.tensor(1.0 / (448.0 * 512.0), dtype=torch.float32)
    )
    return codes.view(torch.uint8).numpy(), scale.numpy()


def _token_rows(hidden: int, kind: str, rows: int = 16, seed: int = 0) -> np.ndarray:
    gen = np.random.default_rng(seed + hidden)
    h = gen.standard_normal((rows, hidden)).astype(F32)
    if kind == "heavy":
        h *= np.exp(gen.standard_normal((rows, hidden))).astype(F32)
    elif kind == "outlier":
        h[:, :: hidden // 8] *= 300
    elif kind == "edges":
        h[0] = 0
        h[1] = 2.0**-130  # BF16 subnormals: amax below the minimum scale
        h[2] = -3.0e38
        h[3, 5] = 1.0e30
        h[4] *= 2.0**-120
    return _bf16(h)


@pytest.mark.parametrize("hidden", [2048, 4096, 7168])
@pytest.mark.parametrize("kind", ["normal", "heavy", "outlier", "edges"])
def test_the_token_fp8_stage_writes_the_per_token_rule(hidden, kind):
    """Every cluster partition of the token gives the codes and scales the
    per-token rule gives for the BF16 output, bit for bit (a maximum does not
    depend on its order)."""
    h = _token_rows(hidden, kind)
    want_codes, want_scale = _triton_rule(h)
    clusters = [c for c in (1, 2, 4, 8) if (hidden // 256) % c == 0]
    for cluster in clusters:
        codes, scale = _fused_token_fp8(h, cluster)
        assert np.array_equal(codes, want_codes), cluster
        assert np.array_equal(scale.view(np.int32), want_scale.view(np.int32)), cluster
    if kind == "edges":
        assert (want_codes[0] == 0).all() and want_scale[0, 0] == F32(
            1.0 / (448.0 * 512.0)
        )
        assert want_scale[1, 0] == F32(1.0 / (448.0 * 512.0))
