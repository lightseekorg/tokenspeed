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
SHAPE_SETS and defaults to K3's TP16 table.
The shapes are the exact (N, K) that the decode path hands the routed GEMV,
extracted from a trace of the serving path (rowcta's launch grid is ``(N,)``,
so gridX identifies each call site). Every measurement cycles
through NUM_COPIES independent weight tensors so the L2 never holds the
operand between calls -- serving streams a different layer's weight each
launch, and a single-tensor benchmark distorts the ranking (hot-L2 numbers at
6288x7168 ran 1.9x faster than the serving trace's, and cold-L2 reproduces the
serving per-shape times within ~5%).

A backend earns a routing entry only by beating the incumbent (the kernel
dispatch picks today) by at least MARGIN.
Emits the MEASURED_ROUTE dict for ops/gemm/routed_gemv.py.
"""

from __future__ import annotations

import json
import sys

import torch

# (N, K, calls/step, label). The (N, K) are OBSERVED -- logged at the
# decode_gemv dispatch entry during a real TP16 bs=1 run, not derived. Two of
# the three differ from the TP8 table, and the TP8 table's largest entry
# (6288x7168) has no TP16 counterpart in decode_gemv at all.
#
# calls/step is NOT measured here: the shape probe dedups, so it reports which
# shapes exist and not how often each fires. It is left at 0 so the per-step
# projection below stays zero rather than quoting a number built on a count
# carried over from a different parallelism. Labels are the shapes themselves;
# mapping them to call sites would be a guess until the counts are traced.
SHAPE_SETS = {
    "k3_tp16": (
        [
            (3584, 7168, 92, "n3584_k7168"),
            (2880, 7168, 0, "n2880_k7168"),
            (1152, 1536, 0, "n1152_k1536"),
            (7168, 1536, 69, "kda_o_proj_shard"),
            (1536, 7168, 92, "shared_gate_up_shard"),
            (7168, 768, 92, "shared_down_shard"),
            (1536, 1536, 12, "dspark_q_b_tp8"),
            (7168, 1024, 12, "dspark_o_proj_tp8"),
            (7168, 1792, 12, "dspark_down_tp8"),
            (768, 1536, 0, "qb_tp16"),
            (1792, 7168, 0, "dspark_gate_up_tp16"),
            (2112, 14336, 0, "eagle3_fused_qkv_a_tp16"),
            (2304, 7168, 0, "eagle3_gate_up_tp16"),
            (7168, 512, 0, "o_proj_tp16"),
            (7168, 896, 0, "dspark_down_tp16"),
            (7168, 1152, 0, "eagle3_down_tp16"),
            (2304, 1536, 0, "mla_q_b"),
            (6288, 7168, 0, "kda_in_proj"),
            (3648, 7168, 0, "mla_fused_qkv_a_gate"),
        ],
        # Table keys on exact M; sweep the routed range with no holes.
        list(range(1, 33)),
    ),
    "qwen38_next_tp4": (
        [
            (512, 2560, 0, "mlp_gate"),
            (320, 2560, 0, "shared_gate_up"),
            (2560, 160, 0, "shared_down"),
            (2560, 1536, 0, "attn_o_proj"),
            (4120, 2560, 0, "linear_attn_in_proj"),
            (3584, 2560, 0, "n3584_k2560"),
            (640, 2560, 0, "n640_k2560"),
            (2560, 2560, 0, "n2560_k2560"),
            (12800, 2560, 0, "n12800_k2560"),
        ],
        # Table keys on exact M; sweep the routed range with no holes. The
        # earlier {1, 2, 4, 8} pass left holes that serving does hit --
        # speculative-verify widths reached 12800x2560 at an unswept M and fell
        # back to cublas.
        list(range(1, 33)),
    ),
}
SHAPES, MS = SHAPE_SETS[sys.argv[1] if len(sys.argv) > 1 else "k3_tp16"]
NUM_COPIES = 8
# 41-round medians repeat within ~1-2%, so 4% clears noise without excluding
# the consistent 6-11% skinny wins.
MARGIN = 1.04
BACKENDS = ("cublas", "rowcta", "skinny", "tgv", "ll_bf16")


def timed(fns, iters: int = 96, rounds: int = 41) -> float:
    """Median us/call; each graph iteration advances to the next weight copy."""
    n = len(fns)
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fns[i % n]()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(iters):
            fns[i % n]()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / iters)
    out.sort()
    return out[len(out) // 2]


def candidates(m: int, n: int, k: int):
    """Yield (name, per-copy callables, sample output, reference) per backend."""
    xs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        for _ in range(NUM_COPIES)
    ]
    ws = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        for _ in range(NUM_COPIES)
    ]
    o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    ref = (xs[0] @ ws[0].t()).float()

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

        bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)

        def tg(i):
            return lambda: mm_bf16(
                xs[i], ws[i].t(), bias=bias, pdl=True, backend="tgv", out=o
            )

        yield "tgv", [tg(i) for i in range(NUM_COPIES)], o, ref
    except ImportError:
        pass

    from tokenspeed_kernel.ops.gemm.ll_bf16 import ll_bf16_mm, ll_bf16_mm_supported

    if ll_bf16_mm_supported(xs[0], ws[0]):

        def ll(i):
            return lambda: ll_bf16_mm(xs[i], ws[i], out=o)

        yield "ll_bf16", [ll(i) for i in range(NUM_COPIES)], o, ref


route: dict[str, str] = {}
per_step_gain: dict[int, float] = dict.fromkeys(MS, 0.0)
print(f"cold-L2 sweep: {NUM_COPIES} weight copies per backend")
print(f"{'call site':<18}{'NxK':>12} {'M':>2}  ", end="")
print("  ".join(f"{t:>8}" for t in BACKENDS), end="")
print(f"  {'winner':<8} {'gain':>6}")
print("-" * 102)
for n, k, calls, label in SHAPES:
    for m in MS:
        times: dict[str, float | None] = {}
        for name, fns, o, ref in candidates(m, n, k):
            try:
                fns[0]()
                torch.cuda.synchronize()
                err = (o.float() - ref).abs().max().item()
                if err > 1.5:
                    times[name] = None
                    continue
                times[name] = timed(fns)
            except Exception:  # noqa: BLE001
                times[name] = None
        ok = {t: v for t, v in times.items() if v is not None and t != "cublas"}
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
            per_step_gain[m] += (incumbent - ok[best]) * calls
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
if any(c for _, _, c, _ in SHAPES):
    # A step runs one M, so the widths are mutually exclusive and never summed.
    for m in MS:
        print(f"projected saving vs incumbent at M={m}: {per_step_gain[m]:.0f} us/step")
else:
    print("per-step projection skipped: calls/step not measured")
print(json.dumps(route, indent=2))
if len(sys.argv) > 2:
    json.dump(route, open(sys.argv[2], "w"), indent=2)
