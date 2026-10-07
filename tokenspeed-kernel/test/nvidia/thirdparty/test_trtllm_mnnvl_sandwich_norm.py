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

"""The pre/post ("sandwich") norm boundary fused into the MNNVL all-reduce,
against the unfused chain on the same workspace and FP64.

The unfused chain is the plain all-reduce on the group's workspace, then
``rmsnorm`` of the sum and ``rmsnorm`` with the residual,
``round_residual_sum_bf16`` and the scale pair (ops/layernorm/triton.py). The
element bounds and scale pairs are the CPU test's
(nvidia/ops/communication/test_allreduce_sandwich_rmsnorm.py) for an order-only
change of the two sums of squares; its mismatch and RMS limits hold for the
calls of one hidden size and input kind together. The fused rows do not depend
on the batch around them, the strategy or the rank.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=4 -m pytest -q <this file>``;
``SANDWICH_NORM_HIDDENS=a,b,...`` replaces the hidden sizes.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

HIDDENS = tuple(
    int(h) for h in os.environ.get("SANDWICH_NORM_HIDDENS", "2048,4096,7168").split(",")
)
MAX_TOKENS = 2048

# The CPU test's element bounds.
RESIDUAL_ADDEND_ULPS = 2
XP_MARGIN = 2.0**-5
NORM_ULPS = 1
# The CPU test's limits, over all calls of one hidden size and input kind.
POOLED_MAX_MISMATCH = 1e-3
POOLED_RMS_RATIO = (0.99, 1.01)
# One call of a few rows: an FP32 ulp of a row's rstd flips several of its
# elements, so a single-row call can differ in a few tenths of a percent of
# its elements, and its RMS ratio rests on few samples.
CALL_MAX_MISMATCH = 0.02
CALL_RMS_RATIO = (0.95, 1.05)
# The CPU test's scale pairs: x_scale is not a power of two, so a dropped
# rounding of a or of x_scale * a changes the bits.
SCALE_PAIRS = ((0.3, 0.97), (1.1, 1.0))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {2, 4, 8, 16},
    reason="launch with torchrun world size 2, 4, 8 or 16",
)


def _setup():
    lrank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lrank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), torch.device("cuda", lrank)


def _armed():
    from tokenspeed_kernel.ops.communication import trtllm

    rank, dev = _setup()
    try:
        ok = trtllm.ensure_workspace_initialized(
            rank=rank,
            group=dist.group.WORLD,
            max_token_num=MAX_TOKENS,
            hidden_dim=max(HIDDENS),
        )
    except RuntimeError as exc:
        pytest.skip(f"trtllm fusion workspace unavailable: {exc}")
    manager = trtllm._manager_for_group(dist.group.WORLD)
    if not ok or manager.mnnvl_workspace is None:
        pytest.skip("no MNNVL workspace on this fabric")
    return rank, dev


def _inputs(rank, dev, tokens, hidden, kind, seed=0):
    gen = torch.Generator(device=dev)
    gen.manual_seed(1000 * seed + 17 * rank + hidden)
    x = torch.randn(tokens, hidden, generator=gen, device=dev)
    if kind == "large":
        x *= 64
    elif kind == "heavy":
        x *= torch.exp(torch.randn(tokens, hidden, generator=gen, device=dev))
    elif kind == "outlier":
        x[:, :: hidden // 8] *= 200
    shared = torch.Generator(device=dev)
    shared.manual_seed(7 + seed + hidden)
    residual = torch.randn(tokens, hidden, generator=shared, device=dev) * 4
    post = 1 + 0.1 * torch.randn(hidden, generator=shared, device=dev)
    weight = (1 + 0.3 * torch.randn(hidden, generator=shared, device=dev)).abs()
    return tuple(t.to(torch.bfloat16).contiguous() for t in (x, residual, post, weight))


EPS = 1e-5


def _fused(x, residual, post, weight, rank, pair, **kwargs):
    # The public entry takes PDL from pdl_enabled() and completes at the end;
    # a test that pins either calls the backend entry it forwards to.
    from tokenspeed_kernel.ops.communication import allreduce_sandwich_rmsnorm, trtllm

    fused = trtllm.allreduce_sandwich_rmsnorm if kwargs else allreduce_sandwich_rmsnorm
    return fused(
        x,
        residual,
        post,
        weight,
        rank,
        dist.group.WORLD,
        eps=EPS,
        x_scale=pair[0],
        residual_scale=pair[1],
        max_token_num=MAX_TOKENS,
        **kwargs,
    )


def _plain_allreduce(x):
    """The plain all-reduce on the group's MNNVL workspace at the strategy the
    fused call resolves: the sum the fused epilogue starts from."""
    from tokenspeed_kernel.ops.communication import trtllm
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        AllReduceFusionPattern,
        trtllm_allreduce_fusion,
    )

    mnnvl = trtllm._manager_for_group(dist.group.WORLD).mnnvl_workspace
    tokens, hidden = x.shape
    out = torch.empty_like(x)
    trtllm_allreduce_fusion(
        allreduce_in=x,
        world_size=dist.get_world_size(),
        world_rank=dist.get_rank(),
        token_num=tokens,
        hidden_dim=hidden,
        workspace_ptrs=mnnvl,
        trigger_completion_at_end=True,
        fp32_acc=False,
        pattern_code=AllReduceFusionPattern.kAllReduce,
        use_oneshot=mnnvl.resolve_use_oneshot(tokens, None, hidden),
        allreduce_out=out,
    )
    return out


def _unfused(x, residual, post, weight, pair):
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

    s = _plain_allreduce(x)
    a = rmsnorm(s, post, EPS)
    out, res = rmsnorm(
        a,
        weight,
        EPS,
        residual=residual,
        round_residual_sum_bf16=True,
        x_scale=pair[0],
        residual_scale=pair[1],
    )
    return out, res, s


def _staged(s, residual, post, weight, pair):
    """FP64 with the boundary's BF16 rounding points."""

    def bf16(v64):
        # FP32, then BF16: the rounding the kernels apply to an FP32 value.
        return v64.float().to(torch.bfloat16)

    def norm(v, gamma, e):
        v64 = v.double()
        rstd = torch.rsqrt(
            v64.square().mean(-1, keepdim=True)
            + float(torch.tensor(e, dtype=torch.float32))
        )
        return bf16(v64 * rstd * gamma.double())

    xs, rs = (1.0, 1.0) if pair[0] is None else pair
    a = norm(s, post, EPS)
    xp = bf16(a.double() * float(torch.tensor(xs, dtype=torch.float32)))
    rp = bf16(residual.double() * float(torch.tensor(rs, dtype=torch.float32)))
    res = bf16(xp.double() + rp.double())
    return norm(res, weight, EPS), res, xp, rp


def _exact(s, residual, post, weight, pair):
    def norm(v, gamma, e):
        return v * torch.rsqrt(v.square().mean(-1, keepdim=True) + e) * gamma.double()

    xs, rs = (1.0, 1.0) if pair[0] is None else pair
    res = norm(s.double(), post, EPS) * xs + residual.double() * rs
    return norm(res, weight, EPS), res


def _ulp(v):
    v = v.double().abs()
    e = torch.floor(torch.log2(torch.where(v > 0, v, torch.ones_like(v))))
    return torch.where(v >= 2.0**-126, torch.exp2(e - 7), torch.full_like(v, 2.0**-133))


def _check_pair(out, res, other_out, other_res, xp, rp, weight):
    """The element bounds and the call's mismatch limit; returns the numbers
    of differing elements of ``res`` and ``out``."""
    mismatches = (
        int((res != other_res).sum().item()),
        int((out != other_out).sum().item()),
    )
    assert max(mismatches) <= CALL_MAX_MISMATCH * res.numel()
    d_res = (res.double() - other_res.double()).abs()
    # A one-ulp flip of a moves x_scale * a by up to two ulps, possibly one
    # binade above the staged xp (the margin); a sum one binade above both
    # addends can round two exact ties to even in opposite directions.
    magnitude = torch.stack(
        [
            res.double().abs(),
            other_res.double().abs(),
            (1 + XP_MARGIN) * xp.double().abs(),
            rp.double().abs(),
        ]
    ).amax(0)
    assert bool((d_res <= RESIDUAL_ADDEND_ULPS * _ulp(magnitude)).all())
    o = other_res.double()
    rstd = torch.rsqrt(o.square().mean(-1, keepdim=True))
    propagated = 1.01 * d_res * rstd * weight.double().abs()
    d_out = (out.double() - other_out.double()).abs()
    mag = torch.maximum(out.double().abs(), other_out.double().abs())
    assert bool((d_out <= NORM_ULPS * _ulp(mag) + propagated).all())
    return mismatches


def _sse(t, ref):
    return (t.double() - ref).square().sum().item()


def _rms_ratio(num, den):
    return 1.0 if num == den == 0.0 else (num / den) ** 0.5


@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("kind", ["normal", "large", "heavy", "outlier"])
def test_fused_boundary_within_the_module_bounds(hidden, kind):
    rank, dev = _armed()
    elements = 0
    mismatches = {"unfused": [0, 0], "staged": [0, 0]}
    sse = {"res": [0.0, 0.0], "out": [0.0, 0.0]}  # fused, unfused vs FP64
    for tokens in (1, 3, 8, 64, 257, 1024):
        x, residual, post, weight = _inputs(rank, dev, tokens, hidden, kind)
        for pair in ((None, None), *SCALE_PAIRS):
            case = (tokens, pair)
            fused = _fused(x, residual, post, weight, rank, pair)
            assert fused is not None, case
            f_out, f_res = fused
            u_out, u_res, s = _unfused(x, residual, post, weight, pair)
            torch.cuda.synchronize()
            ref_out, ref_res, xp, rp = _staged(s, residual, post, weight, pair)
            elements += s.numel()
            for name, (o_out, o_res) in (
                ("unfused", (u_out, u_res)),
                ("staged", (ref_out, ref_res)),
            ):
                counts = _check_pair(f_out, f_res, o_out, o_res, xp, rp, weight)
                mismatches[name] = [a + b for a, b in zip(mismatches[name], counts)]
            exact_out, exact_res = _exact(s, residual, post, weight, pair)
            for name, fused_t, unfused_t, exact_t in (
                ("res", f_res, u_res, exact_res),
                ("out", f_out, u_out, exact_out),
            ):
                num, den = _sse(fused_t, exact_t), _sse(unfused_t, exact_t)
                ratio = _rms_ratio(num, den)
                assert CALL_RMS_RATIO[0] <= ratio <= CALL_RMS_RATIO[1], (case, name)
                sse[name][0] += num
                sse[name][1] += den
            # Every rank holds the same bits.
            for t in (f_out, f_res):
                gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, t)
                assert all(torch.equal(g, t) for g in gathered), case
    for name, counts in mismatches.items():
        assert max(counts) <= POOLED_MAX_MISMATCH * elements, (name, counts)
    for name, (num, den) in sse.items():
        ratio = _rms_ratio(num, den)
        assert POOLED_RMS_RATIO[0] <= ratio <= POOLED_RMS_RATIO[1], (name, ratio)


@pytest.mark.parametrize("hidden", HIDDENS)
def test_rows_do_not_depend_on_the_batch_or_the_strategy(hidden):
    """One row alone (one-shot) and in a batch of 512 (two-shot, another
    cluster shape) gives the same bits: the reduction order is fixed by the
    hidden size."""
    rank, dev = _armed()
    x, residual, post, weight = _inputs(rank, dev, 512, hidden, "heavy", seed=3)
    batch = _fused(x, residual, post, weight, rank, SCALE_PAIRS[0])
    for row in (0, 1, 255, 511):
        sl = slice(row, row + 1)
        alone = _fused(
            x[sl].contiguous(),
            residual[sl].contiguous(),
            post,
            weight,
            rank,
            SCALE_PAIRS[0],
        )
        torch.cuda.synchronize()
        assert torch.equal(alone[0], batch[0][sl]) and torch.equal(
            alone[1], batch[1][sl]
        )


def test_graph_replays_match_eager():
    rank, dev = _armed()
    hidden, tokens = 4096, 8
    x, residual, post, weight = _inputs(rank, dev, tokens, hidden, "normal", seed=5)
    eager = _fused(x, residual, post, weight, rank, SCALE_PAIRS[0])
    static_x, static_r = x.clone(), residual.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _fused(static_x, static_r, post, weight, rank, SCALE_PAIRS[0])
    for seed in range(20):
        nx, nr, _, _ = _inputs(rank, dev, tokens, hidden, "normal", seed=10 + seed)
        static_x.copy_(nx)
        static_r.copy_(nr)
        graph.replay()
        ref = _fused(nx, nr, post, weight, rank, SCALE_PAIRS[0])
        torch.cuda.synchronize()
        assert torch.equal(captured[0], ref[0]) and torch.equal(captured[1], ref[1])
    graph.replay()
    static_x.copy_(x)
    static_r.copy_(residual)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], eager[0]) and torch.equal(captured[1], eager[1])


@pytest.mark.parametrize("tokens", [1, 512])
@pytest.mark.parametrize("pdl", [False, True])
def test_early_pdl_release_gives_the_same_bits(tokens, pdl):
    """trigger_completion_at_end=False releases PDL dependents once the inputs
    are loaded (the gammas' shared-memory copies complete first). One boundary
    and two chained ones, the second reading the first's residual, give the
    bits of the default launch at either strategy."""
    rank, dev = _armed()
    for hidden in HIDDENS:
        x, residual, post, weight = _inputs(rank, dev, tokens, hidden, "heavy", seed=7)
        x2, _, _, _ = _inputs(rank, dev, tokens, hidden, "normal", seed=8)

        def chain(early):
            kwargs = {"launch_with_pdl": pdl, "trigger_completion_at_end": not early}
            out1, res1 = _fused(
                x, residual, post, weight, rank, SCALE_PAIRS[0], **kwargs
            )
            out2, res2 = _fused(x2, res1, post, weight, rank, SCALE_PAIRS[1], **kwargs)
            return out1, res1, out2, res2

        default = chain(early=False)
        early = chain(early=True)
        torch.cuda.synchronize()
        for a, b in zip(early, default):
            assert torch.equal(a, b), (hidden, tokens, pdl)


def test_calls_the_kernel_cannot_serve_return_none():
    rank, dev = _armed()
    x, residual, post, weight = _inputs(rank, dev, 4, 2880, "normal")
    assert _fused(x, residual, post, weight, rank, (None, None)) is None
    x, residual, post, weight = _inputs(rank, dev, MAX_TOKENS + 1, 2048, "normal")
    assert _fused(x, residual, post, weight, rank, (None, None)) is None
