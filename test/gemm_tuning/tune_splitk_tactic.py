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

"""Measured split-K tactics for the DFlash2 draft's GEMM shapes.

flashinfer's ``default_tactic`` is a generic occupancy heuristic; these shapes
are cold-weight and grid-starved, so the pick these shapes need is not the one
it makes. Run as ``tune_splitk_tactic.py [shape_set] [tactic.json]``; the set names
a key of ``_bench.SHAPE_SETS`` and defaults to the TP8 draft. Emits the table
for ops/gemm/, plus the incumbent it has to beat.

``mma_n`` stays inside the vendor's declared ``_SUPPORTED_MMA_N``; the
``_MAX_M`` cutover is widened by ``thirdparty.cute_dsl.flashinfer_splitk``,
whose tests check the kernel is exact past it.
"""

import dataclasses
import itertools
import json
import sys

import flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk as K
import torch
from _bench import MARGIN, NUM_COPIES, operands, select_shape_set, try_time
from flashinfer import mm_bf16
from tokenspeed_kernel.ops.gemm.routed_gemv import _skinny_config
from tokenspeed_kernel.thirdparty.cute_dsl import flashinfer_splitk
from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
    shape_dynamic_skinny_gemm as skinny,
)

assert flashinfer_splitk.is_available(), "not the measured flashinfer build"

SHAPES, BATCHES = select_shape_set(
    sys.argv[1] if len(sys.argv) > 1 else "k3_dflash2_tp8"
)
ITERS, ROUNDS = 64, 21


def reachable(m: int, n: int, k: int, xs, ws, o):
    """The backends ops/gemm/routed_gemv can actually DISPATCH at this shape.

    A backend the router has no wrapper for cannot be the thing a tactic has to
    beat -- scoring against tinygemm, which is unroutable, silently suppresses
    real entries.

    Args:
        m: Rows of the activation.
        n: Rows of the weight.
        k: Reduction extent.
        xs: Activation copies.
        ws: Weight copies.
        o: Shared output buffer.

    Yields:
        ``(name, per-copy callables)`` per dispatchable backend.
    """
    yield "cublas", [
        (lambda i: lambda: torch.mm(xs[i], ws[i].t(), out=o))(i)
        for i in range(NUM_COPIES)
    ]
    yield "tgv", [
        (lambda i: lambda: mm_bf16(xs[i], ws[i].t(), pdl=True, backend="tgv", out=o))(i)
        for i in range(NUM_COPIES)
    ]
    cfg = _skinny_config(m, n, k)
    if skinny.is_available() and skinny.supports(cfg, m, n, k):
        yield "skinny", [
            (lambda i: lambda: skinny(xs[i], ws[i], cfg, out=o))(i)
            for i in range(NUM_COPIES)
        ]
    if m <= 32:
        # ll_bf16 prefers this backend where the wheel carries it.
        yield "ll_bf16", [
            (
                lambda i: lambda: mm_bf16(
                    xs[i], ws[i].t(), pdl=True, backend="cute-dsl", out=o
                )
            )(i)
            for i in range(NUM_COPIES)
        ]


def tactics(m: int, n: int, k: int):
    """Every split-K tactic the vendor validates for this shape.

    Args:
        m: Rows of the activation.
        n: Rows of the weight.
        k: Reduction extent.

    Yields:
        ``SplitKTactic`` instances, each already given its widest legal
        ``ab_stages``.
    """
    for mma_m, mma_n, split_k in itertools.product(
        (64, 128), K._SUPPORTED_MMA_N, (1, 2, 3, 4)
    ):
        tactic = K.SplitKTactic(mma_m, mma_n, split_k, 2)
        try:
            K.validate_tactic(tactic, m, n, k)
        except ValueError:
            continue
        yield dataclasses.replace(
            tactic,
            ab_stages=min(K._max_ab_stages_for(tactic, K._SMEM_CAPACITY_BYTES), 12),
        )


table: dict[str, tuple] = {}
gain = 0.0
print(
    f"{'call site':<18}{'NxK':>12}{'M':>4}{'incumbent':>11}{'tuned':>8}"
    f"{'gain':>7}  tactic"
)
print("-" * 88)
for n, k, calls, per_row, label in SHAPES:
    for batch in BATCHES:
        m = batch * per_row
        xs, ws, o, ref = operands(m, n, k, NUM_COPIES)

        incumbent, incumbent_name = 1e9, "-"
        for name, fns in reachable(m, n, k, xs, ws, o):
            elapsed, _ = try_time(fns, o, ref, ITERS, ROUNDS)
            if elapsed is not None and elapsed < incumbent:
                incumbent, incumbent_name = elapsed, name

        best, best_tactic = 1e9, None
        for tactic in tactics(m, n, k):
            fns = [
                (
                    lambda i, t=tactic: lambda: K.run_splitk_dense(
                        xs[i], ws[i].t(), None, o, True, t
                    )
                )(i)
                for i in range(NUM_COPIES)
            ]
            elapsed, _ = try_time(fns, o, ref, ITERS, ROUNDS)
            if elapsed is not None and elapsed < best:
                best, best_tactic = elapsed, tactic
        if best_tactic is not None and best * MARGIN <= incumbent:
            table[f"{m},{n},{k}"] = dataclasses.astuple(best_tactic)
            gain += (incumbent - best) * calls
            mark = f"{incumbent / best:6.2f}x"
        else:
            mark = "  keep"
        print(
            f"{label:<18}{f'{n}x{k}':>12}{m:>4}{incumbent:>8.2f}"
            f"({incumbent_name[:3]}){best:>8.2f}{mark:>7}  "
            f"{dataclasses.astuple(best_tactic) if best_tactic else '-'}",
            flush=True,
        )
        del xs, ws, o, ref
        torch.cuda.empty_cache()

print(f"\nprojected saving per drafter step, summed over call sites: {gain:.0f} us")
print(json.dumps(table, indent=2))
if len(sys.argv) > 2:
    json.dump(table, open(sys.argv[2], "w"), indent=2)
