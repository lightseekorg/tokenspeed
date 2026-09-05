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
are cold-weight and grid-starved, so the pick that wins here is not the pick it
makes. Emits the table for ops/gemm/, plus the incumbent it has to beat.

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
from flashinfer import mm_bf16
from tokenspeed_kernel.ops.gemm.routed_gemv import _skinny_config
from tokenspeed_kernel.thirdparty.cute_dsl import flashinfer_splitk
from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
    shape_dynamic_skinny_gemm as skinny,
)

assert flashinfer_splitk.is_available(), "not the measured flashinfer build"

# (N, K, calls/step, M-per-batch-row, label) for the TP8 draft.
SHAPES = [
    (1792, 7168, 10, 8, "conv_kernel_proj"),
    (2112, 7168, 5, 8, "fused_qkv_a"),
    (1536, 1536, 5, 8, "q_b_proj"),
    (7168, 1024, 5, 8, "o_proj"),
    (3584, 7168, 5, 8, "gate_up_proj"),
    (7168, 1792, 5, 8, "down_proj"),
    (256, 7168, 1, 7, "selector_proj"),
    (20480, 7168, 1, 7, "lm_head"),
]
BATCHES = [1, 2, 4, 8]
COPIES, ITERS, ROUNDS = 8, 64, 21
MARGIN = 1.04


def timed(fns):
    n = len(fns)
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(ITERS):
            fns[i % n]()
    torch.cuda.synchronize()
    out = []
    for _ in range(ROUNDS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / ITERS)
    out.sort()
    return out[len(out) // 2]


def try_time(fn, xs, ws, o, ref):
    try:
        fn(0)
        torch.cuda.synchronize()
        if (o.float() - ref).abs().max().item() / ref.abs().max().item() > 0.02:
            return None
        return timed([(lambda i: lambda: fn(i))(i) for i in range(COPIES)])
    except Exception:
        return None


table, gain = {}, 0.0
print(
    f"{'call site':<18}{'NxK':>12}{'M':>4}{'incumbent':>11}{'tuned':>8}"
    f"{'gain':>7}  tactic"
)
print("-" * 88)
for n, k, calls, per_row, label in SHAPES:
    for b in BATCHES:
        m = b * per_row
        xs = [
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            for _ in range(COPIES)
        ]
        ws = [
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            for _ in range(COPIES)
        ]
        o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        ref = xs[0].float() @ ws[0].float().t()

        # The incumbent is what ops/gemm/routed_gemv can actually DISPATCH:
        # cuBLAS, tgv, skinny, and ll_bf16 (the cute-dsl default tactic). A
        # backend the router has no wrapper for cannot be the thing this has to
        # beat -- scoring against tinygemm, which wins 1536x1536 but is
        # unroutable, silently suppressed real entries.
        reachable = [
            ("cublas", lambda i: torch.mm(xs[i], ws[i].t(), out=o)),
            (
                "tgv",
                lambda i: mm_bf16(xs[i], ws[i].t(), pdl=True, backend="tgv", out=o),
            ),
        ]
        cfg = _skinny_config(m, n, k)
        if skinny.is_available() and skinny.supports(cfg, m, n, k):
            reachable.append(("skinny", lambda i: skinny(xs[i], ws[i], cfg, out=o)))
        if m <= 32:
            # ll_bf16 prefers this backend where the wheel carries it.
            reachable.append(
                (
                    "ll_bf16",
                    lambda i: mm_bf16(
                        xs[i], ws[i].t(), pdl=True, backend="cute-dsl", out=o
                    ),
                )
            )
        inc, inc_name = 1e9, "-"
        for nm, fn in reachable:
            t = try_time(fn, xs, ws, o, ref)
            if t is not None and t < inc:
                inc, inc_name = t, nm

        best, best_t = 1e9, None
        for mma_m, mma_n, sk in itertools.product(
            (64, 128), K._SUPPORTED_MMA_N, (1, 2, 3, 4)
        ):
            tac = K.SplitKTactic(mma_m, mma_n, sk, 2)
            try:
                K.validate_tactic(tac, m, n, k)
            except ValueError:
                continue
            tac = dataclasses.replace(
                tac,
                ab_stages=min(K._max_ab_stages_for(tac, K._SMEM_CAPACITY_BYTES), 12),
            )
            t = try_time(
                lambda i, tac=tac: K.run_splitk_dense(
                    xs[i], ws[i].t(), None, o, True, tac
                ),
                xs,
                ws,
                o,
                ref,
            )
            if t is not None and t < best:
                best, best_t = t, tac
        if best_t is not None and best * MARGIN <= inc:
            table[f"{m},{n},{k}"] = dataclasses.astuple(best_t)
            gain += (inc - best) * calls
            mark = f"{inc / best:6.2f}x"
        else:
            mark = "  keep"
        print(
            f"{label:<18}{f'{n}x{k}':>12}{m:>4}{inc:>8.2f}({inc_name[:3]}){best:>8.2f}"
            f"{mark:>7}  {dataclasses.astuple(best_t) if best_t else '-'}",
            flush=True,
        )
        del xs, ws, o, ref
        torch.cuda.empty_cache()

print(f"\nprojected saving per drafter step, summed over call sites: {gain:.0f} us")
print(json.dumps(table, indent=2))
if len(sys.argv) > 1:
    json.dump(table, open(sys.argv[1], "w"), indent=2)
