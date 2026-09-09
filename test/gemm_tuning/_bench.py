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

"""Shared cold-L2 benchmark harness for the GEMM tuning scripts.

``tune_route.py`` picks a backend per shape and ``tune_splitk_tactic.py`` picks
a split-K tactic for one of them. They have to agree on the shapes, on how a
call is timed and on what counts as a wrong answer, so all three live here.
"""

from __future__ import annotations

import torch

# (N, K, calls/step, M per batch row, label). The (N, K) are OBSERVED at the
# decode_gemv dispatch entry, not derived. calls/step is 0 where the counts were
# never traced, which zeroes the per-step projection instead of quoting one from
# a different parallelism. M per batch row is what the call site multiplies the
# batch by: 1 for ordinary decode, the block width for a per-block shape.
SHAPE_SETS = {
    "k3_tp16": (
        [
            (3584, 7168, 92, 1, "n3584_k7168"),
            (2880, 7168, 0, 1, "n2880_k7168"),
            (1152, 1536, 0, 1, "n1152_k1536"),
            (7168, 1536, 69, 1, "kda_o_proj_shard"),
            (1536, 7168, 92, 1, "shared_gate_up_shard"),
            (7168, 768, 92, 1, "shared_down_shard"),
            (1536, 1536, 12, 1, "dspark_q_b_tp8"),
            (7168, 1024, 12, 1, "dspark_o_proj_tp8"),
            (7168, 1792, 12, 1, "dspark_down_tp8"),
            (768, 1536, 0, 1, "qb_tp16"),
            (1792, 7168, 0, 1, "dspark_gate_up_tp16"),
            (2112, 14336, 0, 1, "eagle3_fused_qkv_a_tp16"),
            (2304, 7168, 0, 1, "eagle3_gate_up_tp16"),
            (7168, 512, 0, 1, "o_proj_tp16"),
            (7168, 896, 0, 1, "dspark_down_tp16"),
            (7168, 1152, 0, 1, "eagle3_down_tp16"),
            (2304, 1536, 0, 1, "mla_q_b"),
            (6288, 7168, 0, 1, "kda_in_proj"),
            (3648, 7168, 0, 1, "mla_fused_qkv_a_gate"),
        ],
        # Table keys on exact M; sweep the routed range with no holes.
        list(range(1, 33)),
    ),
    # Qwen3.8-Flash-Next BF16 shapes observed at UnquantizedLinearMethod across
    # the FP8/NVFP4 and MTP3/MTP7 serving configurations. Each TP set is tuned
    # independently because tensor-parallel projections change both N and K.
    "qwen38_next_tp4": (
        [
            (512, 2560, 0, 1, "mlp_gate"),
            (320, 2560, 0, 1, "shared_gate_up"),
            (2560, 160, 0, 1, "shared_down"),
            (2560, 1536, 0, 1, "attn_o_proj"),
            (4120, 2560, 0, 1, "linear_attn_in_proj"),
            (3584, 2560, 0, 1, "n3584_k2560"),
            (640, 2560, 0, 1, "n640_k2560"),
            (2560, 2560, 0, 1, "n2560_k2560"),
            (12800, 2560, 0, 1, "n12800_k2560"),
        ],
        list(range(1, 33)),
    ),
    # Kimi-K3 DFlash2 block drafter at TP8. The five draft layers run one row
    # per block position and the selector and head run one per drafted token,
    # so the served M is a batch multiple of 8 or of 7 and never the flat
    # 1..32 the ordinary-decode sets sweep.
    "k3_dflash2_tp8": (
        [
            (1792, 7168, 10, 8, "conv_kernel_proj"),
            (2112, 7168, 5, 8, "fused_qkv_a"),
            (1536, 1536, 5, 8, "q_b_proj"),
            (7168, 1024, 5, 8, "o_proj"),
            (3584, 7168, 5, 8, "gate_up_proj"),
            (7168, 1792, 5, 8, "down_proj"),
            (7168, 35840, 1, 8, "context_fc"),
            (256, 7168, 1, 7, "selector_hidden_proj"),
            (20480, 7168, 1, 7, "lm_head"),
        ],
        list(range(1, 9)),
    ),
    "qwen38_next_tp2": (
        [
            (512, 2560, 0, 1, "n512_k2560"),
            (640, 2560, 0, 1, "n640_k2560"),
            (2560, 320, 0, 1, "n2560_k320"),
            (2560, 2560, 0, 1, "n2560_k2560"),
            (2560, 3072, 0, 1, "n2560_k3072"),
            (6656, 2560, 0, 1, "n6656_k2560"),
            (8240, 2560, 0, 1, "n8240_k2560"),
            (12800, 2560, 0, 1, "n12800_k2560"),
        ],
        list(range(1, 33)),
    ),
}

#: Independent weight copies per backend, so the L2 never holds the operand
#: between calls: serving streams a different layer's weight each launch, and a
#: single-tensor benchmark distorts the ranking.
NUM_COPIES = 8
#: A margin wide enough to sit outside the harness's round-to-round noise.
MARGIN = 1.04
#: BF16 accumulation order differs per kernel; 2% clears that without admitting
#: a kernel that computed the wrong thing. Relative, not absolute: at K=7168 the
#: BF16 output's own rounding is already ~0.4% of a ~400-magnitude entry.
REL_TOL = 0.02


def select_shape_set(name: str) -> tuple[list, list[int]]:
    """Look a shape set up by name.

    Args:
        name: A key of :data:`SHAPE_SETS`.

    Returns:
        ``(shapes, batches)``; multiply a batch by a shape's M-per-row to get
        the M that call site runs.

    Raises:
        SystemExit: The name is not a shape set.
    """
    if name not in SHAPE_SETS:
        raise SystemExit(f"unknown shape set {name!r}; have {sorted(SHAPE_SETS)}")
    return SHAPE_SETS[name]


def operands(m: int, n: int, k: int, copies: int):
    """Cold-L2 operands for one shape.

    Args:
        m: Rows of the activation.
        n: Rows of the weight.
        k: Reduction extent.
        copies: Independent weight/activation pairs to cycle through.

    Returns:
        ``(xs, ws, out, ref)`` -- the copies, one shared output buffer, and the
        fp32 reference for copy 0.
    """
    xs = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for _ in range(copies)]
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(copies)]
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    return xs, ws, out, (xs[0] @ ws[0].t()).float()


def relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Largest elementwise error, as a fraction of the reference's magnitude.

    Args:
        out: The measured output.
        ref: The fp32 reference.

    Returns:
        The relative error; compare against :data:`REL_TOL`.
    """
    return (out.float() - ref).abs().max().item() / ref.abs().max().item()


def timed(fns, iters: int, rounds: int) -> float:
    """Median us/call; each graph iteration advances to the next weight copy.

    Args:
        fns: One callable per weight copy.
        iters: Calls captured into the replayed graph.
        rounds: Replays to take the median over.

    Returns:
        Median microseconds per call.
    """
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


def try_time(fns, out: torch.Tensor, ref: torch.Tensor, iters: int, rounds: int):
    """Time a candidate, or say why it produced no number.

    Args:
        fns: One callable per weight copy.
        out: The buffer ``fns`` write into.
        ref: The fp32 reference for copy 0.
        iters: Calls captured into the replayed graph.
        rounds: Replays to take the median over.

    Returns:
        ``(microseconds, None)`` on success, or ``(None, reason)``.
    """
    try:
        fns[0]()
        torch.cuda.synchronize()
        err = relative_error(out, ref)
        if err > REL_TOL:
            return None, f"wrong result (rel {err:.3f})"
        return timed(fns, iters, rounds), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:70]}"
