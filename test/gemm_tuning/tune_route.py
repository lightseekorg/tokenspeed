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
SHAPE_SETS and defaults to K3's TP8/TP16 shapes.
Shapes are kept explicitly below for the current K3 target and DSpark draft.
Each FI backend is exact-autotuned before comparison; the table records only
the backend, and serving autotunes its tactics on the deployment device.
Every measurement cycles through at least NUM_COPIES independent weights,
covering twice the device's L2 capacity, so the L2 never holds the
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
import math
import sys

import torch
from tokenspeed_kernel.ops.tuning import autotune

# (N, K, label). Omit fused MoE input and old Eagle3 shapes. MLA q_b remains
# because its composed path still reaches GEMM. Shared dimensions appear once;
# no per-step estimates are made from old call counts.
SHAPE_SETS = {
    "k3": (
        [
            (3584, 7168, "dspark_gate_up_tp8"),
            (2880, 7168, "mla_qkv_gate_tp16"),
            (7168, 1536, "attention_o_tp8"),
            (7168, 768, "shared_down_tp8_attn_tp16"),
            (1536, 1536, "dspark_q_b_tp8"),
            (7168, 1024, "dspark_o_proj_tp8"),
            (7168, 1792, "dspark_down_tp8"),
            (768, 1536, "draft_q_b_tp16"),
            (1792, 7168, "dspark_gate_up_tp16"),
            (7168, 512, "draft_o_proj_tp16"),
            (7168, 896, "dspark_down_tp16"),
            (6288, 7168, "kda_in_proj_tp8"),
            (3648, 7168, "mla_qkv_gate_tp8"),
            (3216, 7168, "kda_in_proj_tp16"),
            (7168, 384, "shared_down_tp16"),
            (8448, 7168, "dense_gate_up_tp8"),
            (7168, 4224, "dense_down_tp8"),
            (4224, 7168, "dense_gate_up_tp16"),
            (7168, 2112, "dense_down_tp16"),
            (7168, 35840, "dspark_context_proj"),
            (1152, 1536, "mla_q_b_tp16"),
            (2304, 1536, "mla_q_b_tp8"),
        ],
        # Table keys on exact M; sweep the routed range with no holes.
        list(range(1, 65)),
    ),
    # Qwen3.8-Flash-Next BF16 shapes observed at UnquantizedLinearMethod across
    # the FP8/NVFP4 and MTP3/MTP7 serving configurations. Each TP set is tuned
    # independently because tensor-parallel projections change both N and K.
    "qwen38_next_tp4": (
        [
            (512, 2560, "mlp_gate"),
            (320, 2560, "shared_gate_up"),
            (2560, 160, "shared_down"),
            (2560, 1536, "attn_o_proj"),
            (4120, 2560, "linear_attn_in_proj"),
            (3584, 2560, "n3584_k2560"),
            (640, 2560, "n640_k2560"),
            (2560, 2560, "n2560_k2560"),
            (12800, 2560, "n12800_k2560"),
        ],
        list(range(1, 33)),
    ),
    "qwen38_next_tp2": (
        [
            (512, 2560, "n512_k2560"),
            (640, 2560, "n640_k2560"),
            (2560, 320, "n2560_k320"),
            (2560, 2560, "n2560_k2560"),
            (2560, 3072, "n2560_k3072"),
            (6656, 2560, "n6656_k2560"),
            (8240, 2560, "n8240_k2560"),
            (12800, 2560, "n12800_k2560"),
        ],
        list(range(1, 33)),
    ),
}
SHAPES, MS = SHAPE_SETS[sys.argv[1] if len(sys.argv) > 1 else "k3"]
NUM_COPIES = 8
# 41-round medians repeat within ~1-2%, so 4% clears noise without excluding
# the consistent 6-11% skinny wins.
MARGIN = 1.04
BACKENDS = ("cublas", "rowcta", "skinny", "tgv", "ll_bf16")
CAPTURE_STREAM = torch.cuda.Stream()
torch.manual_seed(0)


def timed(fns, iters: int = 96, rounds: int = 41) -> float:
    """Median us/call; each graph iteration advances to the next weight copy."""
    n = len(fns)
    iters = math.ceil(max(iters, 2 * n) / n) * n
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    s = CAPTURE_STREAM
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fns[i % n]()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
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
    copies = max(
        NUM_COPIES,
        math.ceil(2 * torch.cuda.get_device_properties(0).L2_cache_size / (n * k * 2))
        + 1,
    )
    xs = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for _ in range(copies)]
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(copies)]
    o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    ref = (xs[0] @ ws[0].t()).float()

    def cublas(i):
        return lambda: torch.mm(xs[i], ws[i].t(), out=o)

    yield "cublas", [cublas(i) for i in range(copies)], o, ref

    if m == 1:
        from tokenspeed_kernel.ops.gemm.triton_gemv import rowcta_gemv

        def rc(i):
            return lambda: rowcta_gemv(xs[i], ws[i], o)

        yield "rowcta", [rc(i) for i in range(copies)], o, ref

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

            yield "skinny", [sk(i) for i in range(copies)], o, ref

    try:
        from flashinfer import mm_bf16

        bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)

        def tg(i):
            return lambda: mm_bf16(
                xs[i], ws[i].t(), bias=bias, pdl=True, backend="tgv", out=o
            )

        yield "tgv", [tg(i) for i in range(copies)], o, ref
    except ImportError:
        pass

    from tokenspeed_kernel.ops.gemm.ll_bf16 import ll_bf16_mm, ll_bf16_mm_supported

    if ll_bf16_mm_supported(xs[0], ws[0]):

        def ll(i):
            return lambda: ll_bf16_mm(xs[i], ws[i], out=o)

        yield "ll_bf16", [ll(i) for i in range(copies)], o, ref


route: dict[str, str] = {}
print(f"cold-L2 sweep: at least {NUM_COPIES} weights, covering twice L2")
print(f"{'call site':<26} {'NxK':>12} {'M':>2}  ", end="")
print("  ".join(f"{t:>8}" for t in BACKENDS), end="")
print(f"  {'winner':<8} {'gain':>6}")
print("-" * 102)
for n, k, label in SHAPES:
    for m in MS:
        times: dict[str, float | None] = {}
        for name, fns, o, ref in candidates(m, n, k):
            try:
                if name in ("tgv", "ll_bf16"):
                    with autotune(tuning_buckets=(m,), round_up=False):
                        fns[0]()
                with autotune(tune_mode=False, tuning_buckets=(m,), round_up=False):
                    fns[0]()
                    torch.cuda.synchronize()
                    if not torch.allclose(o.float(), ref, atol=0.5, rtol=2e-2):
                        times[name] = None
                        continue
                    times[name] = timed(fns)
            except Exception as exc:  # noqa: BLE001
                print(f"{(m, n, k)} {name}: {exc}", file=sys.stderr, flush=True)
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
            print(
                f"{label:<26} {f'{n}x{k}':>12} {m:>2}  {cells}  {best:<8} "
                f"{incumbent / ok[best]:5.2f}x"
            )
        else:
            keep = "rowcta" if times.get("rowcta") == incumbent else "cublas"
            print(
                f"{label:<26} {f'{n}x{k}':>12} {m:>2}  {cells}  {keep:<8} "
                f"{'(keep)':>6}"
            )

print()
print(json.dumps(route, indent=2))
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as output:
        json.dump(route, output, indent=2)
