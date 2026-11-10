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

"""Time one Muse Spark TP8 layer launch on 1, 2, 4 or 8 GPUs.

Each GPU runs the TP8 kernel of one rank with the TP8 shapes of that rank.
The kernel keeps RANK and WORLD at 8. Physical GPU p runs logical rank p, and
logical ranks ``--gpus`` to 7 are absent:

* The heap bases of the absent ranks point at a local ghost buffer, so puts
  to an absent rank land there.
* Before each launch, a fill kernel writes the segment of each absent rank in
  every peer region with the tag of this launch, as if that rank had already
  put. All peer regions are rank-major with equal segments. The payloads are
  the segment of rank 0, so sums, norms and routing see realistic values.
* The present ranks still wait for the real puts of each other, over the
  links when ``--gpus`` is more than 1.

The numbers are for iteration, not for claims: 1 GPU has no link latency,
and the absent ranks never arrive late. Confirm a kept change on 8 GPUs.

Run from the repository root (each line is one example):

    PYTHONPATH=tokenspeed-kernel/test HIP_VISIBLE_DEVICES=1 \\
        python tokenspeed-kernel/test/amd/gfx950/ops/monokernel/muse_spark_layer_bench.py --gpus 1
    PYTHONPATH=tokenspeed-kernel/test HIP_VISIBLE_DEVICES=0,1,2,3 \\
        python tokenspeed-kernel/test/amd/gfx950/ops/monokernel/muse_spark_layer_bench.py --gpus 4 \\
        --trace-out /tmp/trace --out-save /tmp/new.pt

The rows' routing picks ``--distinct`` experts (4 a row by default: 32 at 8
rows, the median of served 8-row verifies, which pick 18 to 56). Expert
stages scale with that count, and rows that agree on their experts (the
earlier default, 17 at 8 rows) understate them by about half. The bench
redraws the router bias until the count matches, before it times anything.

The first line of the output is the time per launch. The table after it is the
trace of rank 0: for each stage, the first and last time (us, relative to the
first event) at which a wave starts, has its inputs, has computed, and ends.
``--out-save`` stores outputs of a fixed launch, so two builds can be checked
for bit identity with ``--compare``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import muse_spark_reference as ref  # noqa: E402
import test_muse_spark_layer_gfx950 as layer_test  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import tl, triton  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import plan as plan_mod  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import stuck  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    layer,
    moe_tail,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.qkvg import (  # noqa: E402
    PROJ_WORDS,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

TP, HIDDEN, OWNED, POOL_OWNED, EXPERTS = 8, 8192, 1024, 512, 16
KV_DTYPES = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
# Launches per graph: one per layer copy, so the timings average over them.
LAUNCHES = 40
# The trace buffer: [CTA, 4 points per stage, wave] in 10 ns ticks.
TRACE_SHAPE, TICK_US = (256, 128, 8), 0.01
POINTS = ("start", "ready", "computed", "end")
SAVED_REGIONS = ("lg", "lg2", "ss1", "ss2", "hp", "bq")


@triton.jit(do_not_specialize=["lay", "n"])
def _fill_absent_ranks(
    scratch, idx, bits, step, lay, n, SLOTS: tl.constexpr, BLOCK: tl.constexpr
):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tag = ((tl.load(step).to(tl.int64) * SLOTS + lay + 1) & 0xFFFFFFFF) << 32
    tl.store(
        scratch + tl.load(idx + o, mask=m), tl.load(bits + o, mask=m) | tag, mask=m
    )


def _peer_slices(plan, parity: int):
    """(offset, segment) of copy ``parity`` of each rank-ordered block of
    each peer region (:func:`moe_tail.rank_blocks`), in every row: rank
    ``r`` writes segment ``r`` of a block."""
    return [(off, seg) for _, off, seg in _named_peer_slices(plan, parity)]


def _named_peer_slices(plan, parity: int):
    """:func:`_peer_slices` with each slice's region name first."""
    sizes = dict(plan.regions)
    out = []
    for r in plan.peers:
        blocks = moe_tail.rank_blocks(r, sizes[r])
        assert sum(n for _, n in blocks) == sizes[r], (r, blocks, sizes[r])
        for row in range(plan.rows):
            base = plan.offset(r) + parity * plan.peer_copy(r) + row * plan.row_pairs
            for start, n in blocks:
                assert n % TP == 0, (r, start, n)
                out.append((r, base + start, n // TP))
    return out


def absent_index(plan, parity: int, gpus: int) -> torch.Tensor:
    """Pair indices of the absent ranks' segments in copy ``parity``."""
    return torch.cat(
        [
            torch.arange(off + gpus * seg, off + TP * seg)
            for off, seg in _peer_slices(plan, parity)
        ]
    )


def absent_flags(plan, parity: int, gpus: int) -> torch.Tensor:
    """Which of :func:`absent_index`'s pairs are flags (``Plan.flags``)."""
    return torch.cat(
        [
            torch.full(((TP - gpus) * seg,), plan.is_flag(r))
            for r, _, seg in _named_peer_slices(plan, parity)
        ]
    )


def ghost_payloads(plan, parity: int, gpus: int, payloads=None) -> torch.Tensor:
    """The low words the fill writes into :func:`absent_index`'s pairs:
    ``payloads`` (zeros when None), with every flag pair 1, the payload of
    a raised flag (``Mailbox.poll_flags`` waits for exactly that)."""
    flags = absent_flags(plan, parity, gpus)
    out = torch.zeros(len(flags), dtype=torch.int64) if payloads is None else payloads
    out = out.clone()
    out[flags.to(out.device)] = 1
    return out


def rank0_payloads(plan, parity: int, gpus: int, scratch) -> torch.Tensor:
    """Payloads (low 32 bits) of rank 0's segments, once per absent rank."""
    return (
        torch.cat(
            [
                scratch[off : off + seg].repeat(TP - gpus)
                for off, seg in _peer_slices(plan, parity)
            ]
        )
        & 0xFFFFFFFF
    )


def _distinct(plan, scratch, rows: int) -> int:
    """The distinct experts the last launch's rows picked (every row's 8 ids
    open its route region)."""
    o = plan.offset("route")
    picks = torch.stack(
        [
            scratch[o + r * plan.row_pairs : o + r * plan.row_pairs + 8]
            for r in range(rows)
        ]
    )
    return len(set((picks & 0xFFFFFFFF).flatten().tolist()))


def _operands(
    rank: int,
    length: int,
    rows: int,
    requests: int,
    experts: int,
    bias_std: float,
    route_seed: int,
    kv_dtype,
    dev,
):
    """The launch operands of ``requests`` requests of ``rows`` chained rows
    each (request ``i``'s row 0 at ``length - 73 i``, its own page table)
    whose routing picks among the first ``experts`` experts, the router bias
    drawn with standard deviation ``bias_std`` (seed ``route_seed``), with a
    ``kv_dtype`` KV cache."""
    R = requests * rows
    sh = layer_test._shared(42, length, R)
    # Request i's length, positions and page table: request 0 takes the
    # shared table, the others pages past every earlier request's.
    g = torch.Generator().manual_seed(77)
    lengths = [length - 73 * i for i in range(requests)]
    tables = [sh["table"]]
    for n in lengths[1:]:
        used = (n + rows - 1 + 63) // 64
        first = int(max(int(t.max()) for t in tables)) + 1
        tables.append((first + torch.randperm(used, generator=g)).to(torch.int32))
    pos = torch.cat([torch.arange(n - 1, n - 1 + rows) for n in lengths])
    slots = torch.cat(
        [
            t[q // 64].long() * 64 + q % 64
            for t, q in zip(tables, pos.view(requests, rows))
        ]
    )
    width = max(len(t) for t in tables)
    table = torch.zeros(requests, width, dtype=torch.int32)
    for i, t in enumerate(tables):
        table[i, : len(t)] = t
    end = max(lengths) + rows - 1
    # The caches hold every request's pages (_rank_weights adds 5 pages past
    # its length, so one request keeps the cache it always had).
    w = layer_test._rank_weights(rank, max(end, 64 * (int(table.max()) - 4)))
    # Each row picks 8 of the held experts. The bias spread sets how much
    # the rows agree: the router scores vary by ~0.04 across experts, so a
    # bias spread of 0.1 makes 8 rows pick about the same experts (12-18
    # distinct of 40), and 0.01 about as many as served 8-row verifies do
    # (median 32 distinct; each run prints its count).
    g = torch.Generator().manual_seed(route_seed)
    sh["bias"] = torch.full((256,), -1e4)
    sh["bias"][:experts] = torch.randn(experts, generator=g) * bias_std
    w13, w13s, w2, w2s = ref.preprocessed_experts(rank, experts, dev)
    s13, s2 = w13s.stride(), w2s.stride()
    own = slice(rank * OWNED, (rank + 1) * OWNED)
    own_pool = slice(rank * POOL_OWNED, (rank + 1) * POOL_OWNED)
    h_next = torch.empty(R, OWNED, device=dev)
    a_out = torch.empty(R, HIDDEN, dtype=torch.bfloat16, device=dev)
    attn_ops = (
        bf16_tiles(w["w_qkvg"]).to(dev),
        sh["a_in"].to(dev),
        (pos + 1).to(torch.int32).to(dev),
        slots.to(torch.int32).to(dev),
        table.to(dev),
        w["k_cache"].to(kv_dtype).to(dev),
        w["v_cache"].to(kv_dtype).to(dev),
        ref.rope_table(end + 1).to(dev),
        bf16_tiles(w["w_o"]).to(dev),
        sh["h"][:, own].contiguous().to(dev),
        sh["alpha1"][own].to(dev),
        sh["beta1"][own].to(dev),
        sh["pre_gain"][own].to(dev),
    )
    moe_ops = (
        sh["router"][:, own].contiguous().to(dev),
        sh["bias"].to(dev),
        bf16_tiles(w["w_pool"]).to(dev),
        w13,
        w13s,
        w13.stride(0),
        *s13,
        w2,
        w2s,
        w2.stride(0),
        *s2,
        sh["pre_exp_gain"].to(dev),
        sh["pen_gain"][own_pool].to(dev),
        bf16_tiles(w["w_post"]).to(dev),
        sh["post_gain"].to(dev),
        sh["alpha2"][own].to(dev),
        sh["beta2"].to(dev),
        sh["in_gain"].to(dev),
        h_next,
        a_out,
    )
    return attn_ops, width, moe_ops, h_next, a_out


def _median_us(graph, replays: int) -> float:
    """Median time of one replay, per launch. One disturbed replay does not move it."""
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for i in range(replays):
        start[i].record()
        graph.replay()
        end[i].record()
        # A deep unsynced launch queue (hundreds of replays of LAUNCHES
        # launches each) segfaults the host on gfx950: sync every 16. Each
        # replay has its own events, so the sync's gap is in no replay's time.
        if i % 16 == 15:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    times = sorted(start[i].elapsed_time(end[i]) for i in range(replays))
    dist.barrier()
    return times[replays // 2] * 1e3 / LAUNCHES


def print_trace(t: torch.Tensor, names: list[str]) -> None:
    t = t.double()
    base = t[t > 0].min()
    for i, name in enumerate(names):
        cells = []
        for pt, label in enumerate(POINTS):
            v = t[:, 4 * i + pt, :]
            v = v[v > 0]
            if v.numel():
                lo = (v.min() - base).item() * TICK_US
                hi = (v.max() - base).item() * TICK_US
                cells.append(f"{label} {lo:5.1f}-{hi:5.1f}")
        print(f"  {name:7s} " + " | ".join(cells))


def worker(rank: int, port: int, args) -> None:
    gpus = args.gpus
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=gpus,
        timeout=timedelta(seconds=900),
    )
    dev = torch.device(f"cuda:{rank}")
    ctx = get_or_create_context(heap_size=256 << 20)
    group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
    # With --shift every launch reads the q | k | v projection that the one
    # before it stored, and ends with the next one's (qnext).
    R = args.requests * args.rows
    plan = layer.layer_plan(
        full=args.full,
        rows=R,
        chain=args.rows,
        warm_next=False,
        own_qkvg=not args.shift,
        next_qkvg=args.shift,
        poll_check=args.poll_check,
        poll_timeout_shift=args.poll_timeout_shift,
        check_parity=True,  # as the launches
    )
    proj = torch.zeros(2, R, int(PROJ_WORDS), dtype=torch.int32, device=dev)
    scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
    ghost = torch.zeros_like(scratch)
    bases = torch.empty(TP, dtype=torch.int64, device=dev)
    bases[:gpus] = group.bases.to(dev)
    bases[gpus:] = int(group.bases[rank]) + (ghost.data_ptr() - scratch.data_ptr())
    attn_ops, table_stride, moe_ops, h_next, a_out = _operands(
        rank,
        args.length,
        args.rows,
        args.requests,
        args.experts,
        args.bias_std,
        args.route_seed,
        KV_DTYPES[args.kv_dtype],
        dev,
    )
    step = torch.full((1,), 100, dtype=torch.int32, device=dev)
    idx = [absent_index(plan, p, gpus).to(dev) for p in (0, 1)]
    # The absent ranks' flags read as raised from the first launch on, before
    # the settle passes copy rank 0's payloads.
    bits = [ghost_payloads(plan, p, gpus).to(dev) for p in (0, 1)]

    def fill(lay: int) -> None:
        # The tag of this launch on the absent ranks' slots of copy ``lay & 1``.
        if gpus == TP:
            return
        n = idx[lay & 1].numel()
        _fill_absent_ranks[(triton.cdiv(n, 1024),)](
            scratch,
            idx[lay & 1],
            bits[lay & 1],
            step,
            lay,
            n,
            plan_mod.LAYER_SLOTS,
            1024,
        )

    # 512 MiB: more than L2 and the 256 MB MALL, so a cold launch reads HBM.
    flush = torch.zeros(1 << 29, dtype=torch.int8, device=dev)
    flush_sink = torch.zeros(1, dtype=torch.int64, device=dev)

    def do_flush() -> None:
        if args.flush == "read":
            torch.sum(flush.view(torch.int64), dim=0, keepdim=True, out=flush_sink)
        else:
            flush.zero_()

    trace = torch.zeros(TRACE_SHAPE, dtype=torch.int64, device=dev)
    # The served launches carry a stuck-poll record buffer: so do the bench's.
    # With --stuck-dir it is a file, which outlives a trap, and a sidecar
    # describes the launches for ``muse_spark.stuck_report``.
    path = f"{args.stuck_dir}/rank{rank}.rec" if args.stuck_dir else None
    stuck_rec = stuck.stuck_buffer(256, 8, path, scratch.data_ptr(), rank).tensor
    if path is not None:
        kind = {"full": args.full}
        with open(f"{path}.json", "w") as f:
            json.dump(
                {
                    "layers": {str(i): kind for i in range(LAUNCHES + 1)},
                    "launches": [[R, args.rows]],
                    # The bench's launches alternate the copies and check it.
                    "check_parity": True,
                },
                f,
            )
    group.finish_prepare()

    def launch(lay: int, trace_buf=None) -> None:
        fill(lay)
        layer.launch_layer(
            scratch,
            step,
            lay,
            bases,
            attn_ops,
            table_stride,
            moe_ops,
            None,
            None,
            None,
            proj[lay & 1] if args.shift else None,
            proj[(lay + 1) & 1] if args.shift else None,
            attn_ops[0] if args.shift else None,  # the next layer's: this one's
            trace_buf,
            stuck_rec,
            rank,
            args.full,
            R,
            args.rows,
            False,  # whole_rows: the owned residual columns
            True,  # check_parity: the bench's launches alternate
            False,  # compile_only
        )

    def run(cold: bool) -> None:
        step.add_(1)
        for lay in range(1, LAUNCHES + 1):
            if cold:
                do_flush()
            launch(lay)

    if args.distinct:
        # Redraw the router bias (an operand: no recompile) until the launch's
        # rows pick exactly ``distinct`` experts. Rank 0 decides for all, so
        # the ranks stop at the same seed. A wider spread makes the rows agree
        # (fewer distinct experts): after 40 seeds without a hit, the spread
        # doubles or halves toward the count.
        bias = moe_ops[1]
        std, seed, found = args.bias_std, args.route_seed, False
        for _ in range(25):
            seen = []
            for _ in range(40):
                g = torch.Generator().manual_seed(seed)
                b = torch.full((256,), -1e4)
                b[: args.experts] = torch.randn(args.experts, generator=g) * std
                bias.copy_(b)
                step.add_(1)
                # Layers 1 and 2 by turns: the peer regions' copy (parity)
                # must alternate from one launch to the next (mailbox.Peers).
                # With layer 1 every time, a fast rank's next launch could
                # overwrite the copy a slow rank still polled, and every rank
                # then spun until the fail-stop trap (~1 run in 100).
                lay = 2 - len(seen) % 2
                launch(lay)
                torch.cuda.synchronize()
                u = torch.tensor([_distinct(plan, scratch, R)])
                dist.broadcast(u, 0)
                seen.append(int(u))
                if seen[-1] == args.distinct:
                    found = True
                    break
                seed += 1
            if found:
                break
            std = std * 2 if sum(seen) / len(seen) > args.distinct else std / 2
        if not found:
            raise SystemExit(f"no seed gives {args.distinct} distinct experts")
        if lay == 1:
            # The timed runs start at layer 1: keep the parity alternating.
            step.add_(1)
            launch(2)
            torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(
                f"routing: seed {seed} (bias spread {std:g}) gives "
                f"{args.distinct} distinct experts"
            )

    stream = torch.cuda.Stream()
    graphs = {}
    with torch.cuda.stream(stream):
        if gpus < TP:
            # Values that depend on other ranks' data (the reduced o_proj, its
            # norms) come from launches that read the previous copies. Repeat
            # until both copies hold consistent payloads, or old and new builds
            # give different outputs for no reason.
            for _ in range(args.settle):
                for lay in (1, 2):
                    step.add_(1)
                    launch(lay)
                    torch.cuda.synchronize()
                snapshot = scratch.clone()
                for p in (0, 1):
                    bits[p].copy_(
                        ghost_payloads(
                            plan, p, gpus, rank0_payloads(plan, p, gpus, snapshot)
                        )
                    )
        run(False)
        torch.cuda.synchronize()
        for name, cold in (("warm", False), ("cold", True)):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                run(cold)
            graphs[name] = g
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            step.add_(1)
            for lay in range(1, LAUNCHES + 1):
                do_flush()
        graphs["flush"] = g
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            step.add_(1)
            for lay in range(1, LAUNCHES + 1):
                fill(lay)
        graphs["fill"] = g
    torch.cuda.synchronize()
    dist.barrier()
    us = {name: _median_us(g, args.replays) for name, g in graphs.items()}
    if rank == 0:
        u = _distinct(plan, scratch, R)
        print(f"routing: {u} distinct experts over {R} rows")
        f = us["fill"] if gpus < TP else 0.0
        kind = "full" if args.full else "sliding"
        print(
            f"{gpus} GPU {kind} layer, {args.requests} x {args.rows} rows, "
            f"{args.experts} experts, "
            f"at {args.length} tokens, {args.kv_dtype} KV: "
            f"warm {us['warm'] - f:.1f} us, cold {us['cold'] - us['flush'] - f:.1f} us "
            f"per launch (graph; flush {us['flush']:.1f}, fill {f:.1f} us subtracted)"
        )
    if args.out_save:
        dist.barrier()
        torch.cuda.synchronize()
        step.fill_(500)
        launch(1)
        torch.cuda.synchronize()
        if rank == 0:
            o = plan.offset("route")
            route = (scratch[o : o + 16] & 0xFFFFFFFF).to(torch.int32).cpu()
            regions = {}
            for r, n in plan.regions:
                if r in SAVED_REGIONS:
                    # Launch 1 writes copy 1 of a peer region.
                    off = plan.offset(r) + (plan.peer_copy(r) if r in plan.peers else 0)
                    regions[r] = (scratch[off : off + n] & 0xFFFFFFFF).to(torch.int32)
            torch.save(
                {
                    "h_next": h_next.cpu(),
                    "a_out": a_out.cpu(),
                    "ids": route[:8].clone(),
                    "weights": route[8:].view(torch.float32).clone(),
                    "regions": {r: v.cpu() for r, v in regions.items()},
                },
                args.out_save,
            )
    dist.barrier()
    do_flush()
    torch.cuda.synchronize()
    dist.barrier()
    step.add_(1)
    # After the out-save launch (layer 1) the parity alternates with layer 2.
    launch(2 if args.out_save else 1, trace)
    torch.cuda.synchronize()
    t = trace.cpu()
    names = [s.name for s in plan.stages]
    if args.trace_out:
        torch.save({"t": t, "names": names}, f"{args.trace_out}.{rank}")
    if rank == 0:
        print_trace(t, names)
    dist.barrier()
    dist.destroy_process_group()


def compare(a_path: str, b_path: str) -> bool:
    """Whether two --out-save files hold the same outputs and the same values
    in every scratch region both plans have. A region only one plan has (a
    change that adds or drops one) is printed, not compared."""
    a, b = torch.load(a_path), torch.load(b_path)
    same = all(torch.equal(a[k], b[k]) for k in ("h_next", "a_out", "ids", "weights"))
    ra, rb = a["regions"], b["regions"]
    for name in sorted(set(ra) ^ set(rb)):
        print(f"region {name} only in {'the first' if name in ra else 'the second'}")
    differ = [r for r in ra if r in rb and not torch.equal(ra[r], rb[r])]
    for name in differ:
        print(f"region {name} differs")
    return same and not differ


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), help="present ranks")
    p.add_argument(
        "--flush",
        choices=("read", "write"),
        default="read",
        help="how a cold launch evicts the caches before it",
    )
    p.add_argument("--length", type=int, default=8700, help="tokens in the KV cache")
    p.add_argument("--full", action="store_true", help="a full-attention layer")
    p.add_argument(
        "--kv-dtype", choices=tuple(KV_DTYPES), required=True, help="KV cache dtype"
    )
    p.add_argument(
        "--rows", type=int, default=1, help="chained rows (a verify of rows - 1 drafts)"
    )
    p.add_argument(
        "--requests",
        type=int,
        default=1,
        help="requests a launch, --rows rows each; request i at --length - 73 i "
        "tokens, with its own page table",
    )
    p.add_argument(
        "--experts", type=int, default=EXPERTS, help="experts the routing picks among"
    )
    p.add_argument(
        "--bias-std",
        type=float,
        default=0.01,
        help="router bias spread: 0.01 spreads 8 rows over ~32 of 40 experts "
        "like served verifies; 0.1 makes them agree (12-18)",
    )
    p.add_argument(
        "--distinct",
        type=int,
        default=-1,
        help="redraw the router bias until the rows pick exactly this many "
        "experts; -1: 4 a row, at least 8, at most 3/4 of --experts (served "
        "8-row verifies: median 32, "
        "18-56); 0 keeps --route-seed",
    )
    p.add_argument("--route-seed", type=int, default=7, help="router bias seed")
    p.add_argument(
        "--poll-check",
        type=int,
        default=plan_mod.DEFAULT_POLL_CHECK,
        help="retries between tagged-poll clock checks (compile-time)",
    )
    p.add_argument(
        "--poll-timeout-shift",
        type=int,
        default=plan_mod.DEFAULT_POLL_TIMEOUT_SHIFT,
        help=(
            f"poll timeout: {plan_mod.POLL_TIMEOUT_MANTISSA} << shift ticks at"
            " 100 MHz (compile-time)"
        ),
    )
    p.add_argument("--replays", type=int, default=15)
    p.add_argument(
        "--shift",
        action="store_true",
        help="launches of a run of shifted layers: each reads the q | k | v "
        "projection the launch before it stored and ends with the next one's "
        "(qnext, past one row)",
    )
    p.add_argument(
        "--settle",
        type=int,
        default=16,
        help="passes that make the absent ranks' copies consistent",
    )
    p.add_argument("--trace-out", help="save each rank's trace to PREFIX.<rank>")
    p.add_argument(
        "--stuck-dir",
        help="keep each rank's stuck-poll records in DIR/rank<r>.rec (read them "
        "with muse_spark.stuck_report after a trap)",
    )
    p.add_argument("--out-save", help="save rank 0's outputs of a fixed launch")
    p.add_argument(
        "--view", nargs="+", metavar="TRACE", help="print saved traces and exit"
    )
    p.add_argument(
        "--compare", nargs=2, metavar="OUT", help="check two --out-save files"
    )
    args = p.parse_args()
    if args.distinct < 0:
        # Reachable: 8 rows cannot pick all of few experts.
        rows = args.requests * args.rows
        args.distinct = min(max(8, 4 * rows), max(8, args.experts * 3 // 4))
    if args.view:
        for path in args.view:
            d = torch.load(path)
            print(f"== {path}")
            print_trace(d["t"], d["names"])
        return
    if args.compare:
        print(f"outputs identical: {compare(*args.compare)}")
        return
    if args.gpus is None:
        p.error("--gpus is required to time a layer")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
