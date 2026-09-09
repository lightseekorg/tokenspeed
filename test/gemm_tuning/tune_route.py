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

"""Measure decode-GEMV backends at the served models' real shapes, cold-cache.

Run as ``tune_route.py [shape_set] [route.json]``; the set names a key of
``_bench.SHAPE_SETS`` and defaults to K3's TP16 table.
The shapes are the exact (N, K) that the decode path hands the routed GEMV,
extracted from a trace of the serving path (rowcta's launch grid is ``(N,)``,
so gridX identifies each call site). Every measurement cycles through
NUM_COPIES independent weight tensors so the L2 never holds the operand between
calls, which is the state the serving path hands each shape.

A backend earns a routing entry only by beating the incumbent (the kernel
dispatch picks today) by at least MARGIN.
Emits the MEASURED_ROUTE dict for ops/gemm/routed_gemv.py.
"""

from __future__ import annotations

import collections
import json
import sys

import torch
from _bench import (
    MARGIN,
    NUM_COPIES,
    REL_TOL,
    operands,
    relative_error,
    select_shape_set,
    timed,
)

SHAPES, BATCHES = select_shape_set(sys.argv[1] if len(sys.argv) > 1 else "k3_tp16")
# FlashInfer's BF16 backends, and the ones that refuse pdl=True outright.
FI_BACKENDS = ("cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "cutile", "cute-dsl")
_FI_NO_PDL = frozenset({"cutlass", "cublaslt", "cutile"})
# Only these can be dispatched by ops/gemm/routed_gemv; the rest are measured
# so the field is visible, but cannot be written into the table.
ROUTABLE = frozenset({"skinny", "tgv", "ll_bf16"})
BACKENDS = ("cublas", "rowcta", "skinny", "ll_bf16", *FI_BACKENDS)
ITERS, ROUNDS = 96, 41


def candidates(m: int, n: int, k: int):
    """Yield (name, per-copy callables, sample output, reference) per backend."""
    xs, ws, o, ref = operands(m, n, k, NUM_COPIES)

    def cublas(i):
        return lambda: torch.mm(xs[i], ws[i].t(), out=o)

    yield "cublas", [cublas(i) for i in range(NUM_COPIES)], o, ref

    if m == 1:
        from tokenspeed_kernel.ops.gemm.triton_gemv import rowcta_gemv

        def rc(i):
            return lambda: rowcta_gemv(xs[i], ws[i], o)

        yield "rowcta", [rc(i) for i in range(NUM_COPIES)], o, ref

    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        shape_dynamic_skinny_gemm as skinny,
    )

    if skinny.is_available():
        # Rank the config serving would run, not the bare heuristic, so a
        # re-sweep cannot demote a shape whose win lives in SKINNY_CONFIG_ROUTE.
        from tokenspeed_kernel.ops.gemm.routed_gemv import _skinny_config

        cfg = _skinny_config(m, n, k)
        if skinny.supports(cfg, m, n, k):

            def sk(i):
                return lambda: skinny(xs[i], ws[i], cfg, out=o)

            yield "skinny", [sk(i) for i in range(NUM_COPIES)], o, ref

    try:
        from flashinfer import mm_bf16
    except ImportError:
        mm_bf16 = None
    if mm_bf16 is not None:
        # Only TGV needs a bias operand; routed_gemv keeps a zero one for it.
        # Handing that same bias to the others would measure a shape serving
        # never asks for.
        bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
        # Every backend the wheel declares, each at the PDL setting it accepts.
        # cutlass/cublaslt/cutile REJECT pdl=True rather than ignoring it, so
        # asking them the way tgv is asked scores a raised exception as a loss
        # and drops them from the field silently, so the others are asked with
        # pdl=False rather than skipped.
        for be in FI_BACKENDS:
            pdl = be not in _FI_NO_PDL
            operand = bias if be == "tgv" else None

            def fi(i, be=be, pdl=pdl, operand=operand):
                return lambda: mm_bf16(
                    xs[i], ws[i].t(), bias=operand, pdl=pdl, backend=be, out=o
                )

            yield be, [fi(i) for i in range(NUM_COPIES)], o, ref

    from tokenspeed_kernel.ops.gemm.ll_bf16 import ll_bf16_mm, ll_bf16_mm_supported

    if ll_bf16_mm_supported(xs[0], ws[0]):

        def ll(i):
            return lambda: ll_bf16_mm(xs[i], ws[i], out=o)

        yield "ll_bf16", [ll(i) for i in range(NUM_COPIES)], o, ref


route: dict[str, str] = {}
refusals: dict[str, set[str]] = collections.defaultdict(set)
per_step_gain: dict[int, float] = dict.fromkeys(BATCHES, 0.0)
print(f"cold-L2 sweep: {NUM_COPIES} weight copies per backend")
print(f"{'call site':<18}{'NxK':>12} {'M':>2}  ", end="")
print("  ".join(f"{t:>8}" for t in BACKENDS), end="")
print(f"  {'winner':<8} {'gain':>6}")
print("-" * 102)
for n, k, calls, per_row, label in SHAPES:
    for batch in BATCHES:
        m = batch * per_row
        times: dict[str, float | None] = {}
        for name, fns, o, ref in candidates(m, n, k):
            try:
                fns[0]()
                torch.cuda.synchronize()
                err = relative_error(o, ref)
                if err > REL_TOL:
                    times[name] = None
                    refusals[name].add(f"wrong result (rel {err:.3f})")
                    continue
                times[name] = timed(fns, ITERS, ROUNDS)
            except Exception as exc:  # noqa: BLE001
                # Record why, so a backend that never competes is a decision
                # rather than a blank cell. See the legend printed at the end.
                times[name] = None
                refusals[name].add(f"{type(exc).__name__}: {str(exc)[:70]}")
        ok = {
            t: v
            for t, v in times.items()
            if v is not None and t != "cublas" and t in ROUTABLE
        }
        best = min(ok, key=ok.get) if ok else None
        cells = "  ".join(
            f"{times.get(t):8.3f}" if isinstance(times.get(t), float) else f"{'-':>8}"
            for t in BACKENDS
        )
        # An entry must beat the incumbent selection, not just cuBLAS.
        incumbent = min(
            v for t, v in times.items() if v is not None and t in ("cublas", "rowcta")
        )
        if best and ok[best] * MARGIN <= incumbent:
            route[f"{m},{n},{k}"] = best
            per_step_gain[batch] += (incumbent - ok[best]) * calls
            print(
                f"{label:<18}{f'{n}x{k}':>12} {m:>2}  {cells}  {best:<8} "
                f"{incumbent / ok[best]:5.2f}x"
            )
        else:
            keep = "rowcta" if times.get("rowcta") == incumbent else "cublas"
            print(
                f"{label:<18}{f'{n}x{k}':>12} {m:>2}  {cells}  {keep:<8} "
                f"{'(keep)':>6}"
            )

print()
if refusals:
    print("backends that never produced a number, and why:")
    for name in sorted(refusals):
        for why in sorted(refusals[name]):
            print(f"  {name:<10} {why}")
    print()
if any(c for _, _, c, _, _ in SHAPES):
    # A step runs one batch, and every call site in it fires at that batch.
    for batch in BATCHES:
        print(
            f"projected saving vs incumbent at batch={batch}: "
            f"{per_step_gain[batch]:.0f} us/step"
        )
else:
    print("per-step projection skipped: calls/step not measured")
print(json.dumps(route, indent=2))
if len(sys.argv) > 2:
    json.dump(route, open(sys.argv[2], "w"), indent=2)
