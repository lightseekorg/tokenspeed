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

"""The Inkling2 mid local layer as one launch on four ranks.

The stage tests check each stage's math against torch. Here the whole layer
runs as one persistent launch and, on a copy of every input, as one launch a
stage (slices of the plan, in order). Both must give the same bits: the
composed launch adds only scheduling, so any difference is a placement,
region or argument bug."""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    layer,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    plan as ip,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD = 4
PAGE = 16
STEP = 21
LAYER = 3
# A boundary layer's stored blocks (the embedding and two blocks).
BLOCKS = 3


def _plan(d, rows, chain, experts, boundary, layer0, full_splits):
    if full_splits:
        return ip.full_layer_plan(d, rows, chain, experts, full_splits)
    return ip.local_layer_plan(d, rows, chain, experts, boundary, layer0)


def _kv_proj_tiles(d, qkvr, full):
    """The K and V source projection tiles (as the op packs them): each
    source's value rows of the projection, and a full layer's tap rows."""
    dim, taps, k0 = d.kv_dim, 4 * d.kv_heads, d.k_off
    return tuple(
        bf16_tiles(
            torch.nn.functional.pad(
                torch.cat(
                    (
                        qkvr[k0 + i * dim : k0 + (i + 1) * dim],
                        (
                            qkvr[d.taps_off + i * taps : d.taps_off + (i + 1) * taps]
                            if full
                            else qkvr[:0]
                        ),
                    )
                ),
                (0, 0, 0, d.kv_proj_padded - (d.kv_proj if full else dim)),
            )
        )
        for i in range(2)
    )


def _ring_rows(chain):
    """The conv ring's rows: the taps' history and a verify block (a server
    of 8-token blocks for the decode launches of one row)."""
    return 3 + (8 if chain == 1 else chain)


def _inputs(
    d,
    rank,
    rows,
    chain,
    first,
    experts,
    device,
    padded,
    boundary,
    layer0,
    full_splits,
    n_terms,
):
    """Random layer inputs of ``rank`` with ``experts`` (NVFP4 or MXFP8)
    routed experts: replicated state and weights from one seed, the rank's
    shards and caches from another. The requests in
    ``padded`` are a padded graph batch's padding (``docs/design/
    unified_path.md``): no ring slot (-1), the null page 0 as their pages and
    write slots, and every row at position 0 (``Inkling2Monokernel``). A
    full layer (``full_splits``) takes three attention queries, the K and V
    sources' projections and the rows' tau; ``n_terms`` pending terms (1
    when the draft taps' ``residual_sum`` folded them)."""
    shared = torch.Generator().manual_seed(1)
    own = torch.Generator().manual_seed(100 + rank)
    H, LOCAL = d.hidden, d.local_experts

    def rnd(*shape, g=own, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(device)

    terms = tuple(rnd(rows, H, g=shared).bfloat16() for _ in range(n_terms))
    # A boundary layer's stored blocks (its own block ``stored`` stays for
    # the launch to write; layer 0 stores none before its own) and the new
    # block set's score weights.
    stored = 0 if layer0 else BLOCKS
    blocks = torch.zeros(BLOCKS + 1, rows, H, dtype=torch.bfloat16, device=device)
    blocks[:stored] = rnd(BLOCKS, rows, H, g=shared).bfloat16()[:stored]
    set_weights = rnd(ip.SET_QUERIES, H, g=shared, scale=H**-0.5)

    def query():
        w = rnd(H, g=shared, scale=H**-0.5)
        gain = (torch.rand(H, generator=shared) + 0.5).bfloat16().to(device)
        acc = rnd(rows, H, g=shared)
        m = rnd(rows, g=shared)
        s = (torch.rand(rows, generator=shared) + 1.0).to(device)
        return (w, gain, acc, m, s)

    attn_query = tuple(query() for _ in range(3)) if full_splits else query()
    queries = [attn_query, query()]
    qkvr = rnd(d.full_qkvr if full_splits else d.qkvr, H, scale=1 / 64).bfloat16()
    qkvr_w = bf16_tiles(
        torch.nn.functional.pad(qkvr, (0, 0, 0, d.qkvr_padded - qkvr.shape[0]))
    )
    kv_proj = _kv_proj_tiles(d, qkvr, True) if full_splits else ()
    # Past the scaling floor tau grows with the position; below it, 1.
    tau = (
        (1.0 + torch.rand(rows, generator=shared) * 0.25).to(device)
        if full_splits
        else None
    )
    head_weights = (
        (torch.rand(d.q_heads, 128, generator=own) + 0.5).bfloat16().to(device),
        (torch.rand(d.q_heads, 16, generator=own) + 0.5).bfloat16().to(device),
        rnd(16, 1024).bfloat16(),
        rnd(256, 4, scale=0.5).bfloat16(),
        rnd(256, 4, scale=0.5).bfloat16(),
    )
    wo_w = bf16_tiles(rnd(H, d.q_dim, scale=1 / 32).bfloat16())
    attn_taps = rnd(H, 4, g=shared, scale=0.5).bfloat16()
    mlp_taps = rnd(H, 4, g=shared, scale=0.5).bfloat16()
    router_w = bf16_tiles(rnd(d.routed, H, g=shared, scale=1 / 64).bfloat16())
    router_bias = rnd(d.routed, g=shared, scale=0.1)

    def bank(n, k):
        if experts == "mxfp8":
            codes = torch.randint(0, 256, (LOCAL, n, k), dtype=torch.uint8)
            # Finite E4M3 codes (0x7F and 0xFF are NaNs), small exponents.
            codes = torch.where((codes & 0x7F) == 0x7F, codes ^ 1, codes)
            exponents = torch.randint(112, 118, (LOCAL, n, k // 32), dtype=torch.uint8)
            return codes.view(torch.float8_e4m3fn).to(device), exponents.to(device)
        codes = torch.randint(
            0, 256, (LOCAL, n, k // 2), dtype=torch.uint8, device=device
        )
        scales = (torch.rand(LOCAL, n, k // 16, device=device) * 0.5 + 0.1).to(
            torch.float8_e4m3fn
        )
        wa = (torch.rand(LOCAL, generator=own) + 0.5).to(device)
        return codes, scales, wa, torch.tensor([2.0], device=device)

    up, down = bank(2 * d.inter, H), bank(H, d.inter)
    # Each request has its own length, pages and ring slot.
    requests = rows // chain
    firsts = [first + 37 * q for q in range(requests)]
    pages = (max(firsts) + chain + PAGE - 1) // PAGE
    table = (
        (torch.randperm(2 * requests * pages, generator=own)[: requests * pages] + 1)
        .int()
        .view(requests, pages)
        .to(device)
    )
    slots = 2 * requests * pages * PAGE + PAGE
    k_cache = rnd(slots, 2, 128).bfloat16()
    v_cache = rnd(slots, 2, 128).bfloat16()
    positions = torch.tensor(
        [firsts[r // chain] + r % chain for r in range(rows)],
        dtype=torch.int64,
        device=device,
    )
    for q in padded:
        table[q] = 0
        positions[q * chain : (q + 1) * chain] = 0
    request_of = torch.arange(rows, device=device) // chain
    locations = (
        (table.long()[request_of, positions // PAGE] * PAGE + positions % PAGE)
        .int()
        .to(device)
    )
    ring_rows, width, c0 = _ring_rows(chain), d.ring_width, d.attn_channel0
    ring = rnd(requests + 2, ring_rows, width).bfloat16()
    # The ATTN and MLP streams are replicated state: the same on every rank.
    ring[:, :, c0:] = rnd(requests + 2, ring_rows, width - c0, g=shared).bfloat16()
    ring_slots = (
        (torch.randperm(requests + 1, generator=shared)[:requests] + 1).int().to(device)
    )
    for q in padded:
        ring_slots[q] = -1
    return {
        "dims": d,
        "qkvr": qkvr,
        "boundary": boundary,
        "first": layer0,
        "stored": stored,
        "full_splits": full_splits,
        "kv_proj": kv_proj,
        "tau": tau,
        "blocks": blocks,
        "block_count": torch.tensor([stored], dtype=torch.int32, device=device),
        "set_weights": set_weights,
        "terms": terms,
        "queries": tuple(queries),
        "attention_weights": (qkvr_w, head_weights, wo_w, attn_taps),
        "mlp_weights": (router_w, router_bias, up, down, mlp_taps),
        "k_cache": k_cache,
        "v_cache": v_cache,
        "locations": locations,
        "positions": positions,
        "table": table,
        "pages": pages,
        "ring": ring,
        "ring_slots": ring_slots,
    }


def _launch(
    plan, experts, rank, scratch, group, inputs, state, step, parity=0, sources=()
):
    """One launch of ``plan`` on ``state`` (the mutable caches and outputs);
    ``sources`` are a later pass's part's normalized inputs."""
    boundary = inputs["boundary"]
    queries = inputs["queries"]
    sweep = ()
    outputs = state["outputs"]
    if boundary:
        m, s, acc = state["sweep"]
        sweep = (
            state["blocks"],
            inputs["block_count"],
            inputs["set_weights"],
            m,
            s,
            acc,
        )
        # The MLP query's set partial is the sweep's first query.
        w, gain = queries[1][:2]
        queries = (queries[0], (w, gain, acc[0], m[0], s[0]))
        outputs = (state["blocks"][inputs["stored"]],) + outputs[1:]
    layer.launch_gluon_inkling2_local_layer_gfx950(
        plan.key,
        sources,
        experts,
        boundary,
        inputs["first"],
        inputs["full_splits"],
        rank,
        scratch,
        step,
        LAYER,
        parity,
        group.bases,
        inputs["terms"],
        outputs,
        queries,
        sweep,
        inputs["attention_weights"],
        inputs["kv_proj"],
        inputs["tau"],
        (
            (state["k_cache"], state["v_cache"], inputs["locations"]),
            (
                state["k_cache"],
                state["v_cache"],
                inputs["positions"],
                inputs["table"],
                inputs["table"].shape[1],
                PAGE,
            ),
        ),
        (state["ring"], inputs["ring_slots"], inputs["positions"]),
        inputs["mlp_weights"],
        None,
    )


def _state(inputs, rows, device):
    q = ip.SET_QUERIES
    H = inputs["dims"].hidden
    return {
        "outputs": tuple(
            torch.zeros(rows, H, dtype=torch.bfloat16, device=device) for _ in range(3)
        ),
        "blocks": inputs["blocks"].clone(),
        "sweep": (
            torch.zeros(q, rows, dtype=torch.float32, device=device),
            torch.zeros(q, rows, dtype=torch.float32, device=device),
            torch.zeros(q, rows, H, dtype=torch.float32, device=device),
        ),
        "k_cache": inputs["k_cache"].clone(),
        "v_cache": inputs["v_cache"].clone(),
        "ring": inputs["ring"].clone(),
    }


def _worker(
    rank, port, d, rows, chain, first, experts, boundary, layer0, full_splits, n_terms
):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=600),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=512 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        plan = _plan(d, rows, chain, experts, boundary, layer0, full_splits)
        full_scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        slice_scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        inputs = _inputs(
            d,
            rank,
            rows,
            chain,
            first,
            experts,
            device,
            (),
            boundary,
            layer0,
            full_splits,
            n_terms,
        )
        step = torch.full((1,), STEP, dtype=torch.int32, device=device)
        full = _state(inputs, rows, device)
        dist.barrier()
        _launch(plan, experts, rank, full_scratch, group, inputs, full, step)
        torch.cuda.synchronize()
        dist.barrier()
        sliced = _state(inputs, rows, device)
        for stage in plan.stages:
            _launch(
                plan.only((stage.name,)),
                experts,
                rank,
                slice_scratch,
                group,
                inputs,
                sliced,
                step,
            )
        torch.cuda.synchronize()
        dist.barrier()
        # P is the rounded sum of the previous terms (one term as it is).
        total = inputs["terms"][0].float()
        for t in inputs["terms"][1:]:
            total = total + t.float()
        p_out = full["blocks"][inputs["stored"]] if boundary else full["outputs"][0]
        assert torch.equal(p_out, total.bfloat16())
        names = ("P", "A", "M")
        for name, a, b in zip(names, full["outputs"], sliced["outputs"]):
            assert torch.equal(a, b), f"rank {rank}: {name} differs"
            assert torch.isfinite(a.float()).all(), f"rank {rank}: {name} not finite"
        for key in ("k_cache", "v_cache", "ring", "blocks"):
            assert torch.equal(full[key], sliced[key]), f"rank {rank}: {key} differs"
        if boundary:
            for a, b in zip(full["sweep"], sliced["sweep"]):
                assert torch.equal(a, b), f"rank {rank}: set partials differ"
        # A and M are replicated: every rank computes the same bits.
        for t in full["outputs"][1:]:
            gathered = [torch.empty_like(t).cpu() for _ in range(WORLD)]
            dist.all_gather_object(gathered, t.cpu())
            for other in gathered[1:]:
                assert torch.equal(gathered[0], other)
        dist.barrier()
    finally:
        dist.destroy_process_group()


# (boundary, layer 0, full splits, pending terms) of the layer kinds: mid
# and boundary local layers, layer 0 (its one term the embedding), and full
# layers with the draft tap's one folded term (layers 5 .. 41) or three
# (layer 47).
KINDS = [
    (False, False, 0, 3),
    (True, False, 0, 3),
    (True, True, 0, 1),
    (False, False, ip.FULL_SPLITS, 1),
    (False, False, ip.FULL_SPLITS, 3),
]


# Each geometry's launches: Inkling 2 Small's verify blocks of 8, the
# half-width geometry's of 4 (and of 8).
LAUNCHES = [
    *(
        (ip.GEOMETRIES[0], rows, chain, first)
        for rows, chain, first in [
            (1, 1, 50),
            (8, 8, 1500),
            (16, 8, 1500),
            (32, 8, 900),
            (64, 8, 1500),
        ]
    ),
    *(
        (ip.GEOMETRIES[1], rows, chain, first)
        for rows, chain, first in [
            (1, 1, 50),
            (4, 4, 1500),
            (8, 4, 900),
            (16, 4, 1500),
            (32, 4, 1500),
            (64, 8, 1500),
        ]
    ),
]


def _id(value):
    if isinstance(value, ip.Dims):
        return f"h{value.hidden}"
    if isinstance(value, tuple):  # padded requests
        return f"padded{'_'.join(map(str, value))}"
    return str(value)


def _with_kinds(cases):
    """Every case with every layer kind, the experts' format alternating so
    that each case and each kind run with both formats (the expert stages
    do not depend on the kind)."""
    return [
        pytest.param(
            *case,
            experts,
            *kind,
            id="-".join(map(_id, (*case, experts, *kind))),
        )
        for i, case in enumerate(cases)
        for j, kind in enumerate(KINDS)
        for experts in [ip.EXPERT_FORMATS[(i + j) % 2]]
    ]


@pytest.mark.parametrize(
    "d,rows,chain,first,experts,boundary,layer0,full_splits,n_terms",
    _with_kinds(LAUNCHES),
)
def test_local_layer_launch_equals_its_stage_slices(
    d, rows, chain, first, experts, boundary, layer0, full_splits, n_terms
):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _worker,
        args=(
            port,
            d,
            rows,
            chain,
            first,
            experts,
            boundary,
            layer0,
            full_splits,
            n_terms,
        ),
        nprocs=WORLD,
        join=True,
    )


def _request(inputs, q, chain):
    """The inputs of request ``q`` alone: its rows, its page-table row and
    its ring slot (weights and caches shared)."""
    rows = slice(q * chain, (q + 1) * chain)
    one = dict(inputs)
    one["terms"] = tuple(t[rows] for t in inputs["terms"])

    def one_query(query):
        w, gain, acc, m, s = query
        return (w, gain, acc[rows], m[rows], s[rows])

    attn, mlp = inputs["queries"]
    attn = (
        tuple(one_query(x) for x in attn) if inputs["full_splits"] else one_query(attn)
    )
    one["queries"] = (attn, one_query(mlp))
    if inputs["tau"] is not None:
        one["tau"] = inputs["tau"][rows]
    one["blocks"] = inputs["blocks"][:, rows].contiguous()
    one["positions"] = inputs["positions"][rows]
    one["locations"] = inputs["locations"][rows]
    one["table"] = inputs["table"][q : q + 1]
    one["ring_slots"] = inputs["ring_slots"][q : q + 1]
    return one


def _requests_worker(
    rank,
    port,
    d,
    rows,
    chain,
    first,
    padded,
    experts,
    boundary,
    layer0,
    full_splits,
    n_terms,
):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=600),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=512 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        plan = _plan(d, rows, chain, experts, boundary, layer0, full_splits)
        one_plan = _plan(d, chain, chain, experts, boundary, layer0, full_splits)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        one_scratch = ctx.zeros((one_plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        inputs = _inputs(
            d,
            rank,
            rows,
            chain,
            first,
            experts,
            device,
            padded,
            boundary,
            layer0,
            full_splits,
            n_terms,
        )
        full = _state(inputs, rows, device)
        dist.barrier()
        step = torch.full((1,), STEP, dtype=torch.int32, device=device)
        _launch(plan, experts, rank, scratch, group, inputs, full, step)
        torch.cuda.synchronize()
        dist.barrier()
        # The live requests one at a time, on one copy of the caches: their
        # cache rows and ring slots are disjoint.
        alone = _state(inputs, rows, device)
        live = [q for q in range(rows // chain) if q not in padded]
        for i, q in enumerate(live):
            one = _request(inputs, q, chain)
            out = _state(one, chain, device)
            out.update({k: alone[k] for k in ("k_cache", "v_cache", "ring")})
            step = torch.full((1,), STEP + 1 + i, dtype=torch.int32, device=device)
            _launch(one_plan, experts, rank, one_scratch, group, one, out, step, i % 2)
            torch.cuda.synchronize()
            dist.barrier()
            rows_q = slice(q * chain, (q + 1) * chain)
            for name, a, b in zip(("P", "A", "M"), full["outputs"], out["outputs"]):
                assert torch.equal(a[rows_q], b), f"rank {rank}: {name} of {q}"
            if boundary:
                assert torch.equal(full["blocks"][:, rows_q], out["blocks"]), q
                for a, b in zip(full["sweep"], out["sweep"]):
                    assert torch.equal(a[:, rows_q], b), f"rank {rank}: partials of {q}"
        # Padding writes only the null page and no ring slot.
        for key in ("k_cache", "v_cache"):
            assert torch.equal(full[key][PAGE:], alone[key][PAGE:]), key
        assert torch.equal(full["ring"], alone["ring"])
        dist.barrier()
    finally:
        dist.destroy_process_group()


# Each geometry's batches of verify blocks, some of them padding: the live
# requests check the rows contract, the padded ones the padding's.
@pytest.mark.parametrize(
    "d,chain,rows,padded,experts,boundary,layer0,full_splits,n_terms",
    _with_kinds(
        [
            *(
                (ip.GEOMETRIES[0], 8, rows, padded)
                for rows, padded in [(16, (1,)), (32, (2, 3)), (64, (5, 6, 7))]
            ),
            *(
                (ip.GEOMETRIES[1], 4, rows, padded)
                for rows, padded in [(8, (1,)), (16, (2, 3)), (32, (5, 6, 7))]
            ),
        ]
    ),
)
def test_rows_equal_one_request_launches(
    d, chain, rows, padded, experts, boundary, layer0, full_splits, n_terms
):
    """The rows contract (``docs/design/monokernel.md``): a launch of
    ``rows / chain`` verify blocks equals one launch a request bit for bit,
    and a padded request leaves the live requests' caches and ring slots
    alone."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _requests_worker,
        args=(
            port,
            d,
            rows,
            chain,
            1500,
            padded,
            experts,
            boundary,
            layer0,
            full_splits,
            n_terms,
        ),
        nprocs=WORLD,
        join=True,
    )


def _part_worker(rank, port, d, rows, chain, first, experts, full_splits, part):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=600),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=512 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        plan = ip.later_pass_plan(d, rows, chain, experts, full_splits or None, part)
        full_scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        slice_scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        inputs = _inputs(
            d,
            rank,
            rows,
            chain,
            first,
            experts,
            device,
            (),
            False,
            False,
            full_splits,
            1,
        )
        # A later pass projects a local layer's K and V sources too.
        inputs["kv_proj"] = _kv_proj_tiles(d, inputs["qkvr"], bool(full_splits))
        g = torch.Generator().manual_seed(7)
        count = 3 if part == "attention" else 1
        sources = tuple(
            (torch.randn(rows, d.hidden, generator=g) * 2).bfloat16().to(device)
            for _ in range(count)
        )
        step = torch.full((1,), STEP, dtype=torch.int32, device=device)
        full = _state(inputs, rows, device)
        dist.barrier()
        _launch(
            plan, experts, rank, full_scratch, group, inputs, full, step, 0, sources
        )
        torch.cuda.synchronize()
        dist.barrier()
        sliced = _state(inputs, rows, device)
        for stage in plan.stages:
            _launch(
                plan.only((stage.name,)),
                experts,
                rank,
                slice_scratch,
                group,
                inputs,
                sliced,
                step,
                0,
                # Only the first stage reads the part's inputs.
                sources if stage is plan.stages[0] else (),
            )
        torch.cuda.synchronize()
        dist.barrier()
        # The attention part writes A, the MoE part M, and nothing else of
        # the layer's terms.
        written = 1 if part == "attention" else 2
        for i, (a, b) in enumerate(zip(full["outputs"], sliced["outputs"])):
            assert torch.equal(a, b), f"rank {rank}: output {i} differs"
            if i == written:
                assert torch.isfinite(a.float()).all() and a.abs().sum() > 0
            else:
                assert not a.any(), f"rank {rank}: the {part} part wrote output {i}"
        for key in ("k_cache", "v_cache", "ring"):
            assert torch.equal(full[key], sliced[key]), f"rank {rank}: {key} differs"
        changed = not torch.equal(full["k_cache"], inputs["k_cache"])
        assert changed == (part == "attention"), "the KV write belongs to attention"
        gathered = [
            torch.empty_like(full["outputs"][written]).cpu() for _ in range(WORLD)
        ]
        dist.all_gather_object(gathered, full["outputs"][written].cpu())
        for other in gathered[1:]:
            assert torch.equal(gathered[0], other)
        dist.barrier()
    finally:
        dist.destroy_process_group()


# The attention part at either attention kind (it runs no expert stage),
# the MoE part with either experts' format (it does not depend on the
# attention kind).
@pytest.mark.parametrize(
    "part,full_splits,experts",
    [
        ("attention", 0, "nvfp4"),
        ("attention", ip.FULL_SPLITS, "nvfp4"),
        ("mlp", 0, "nvfp4"),
        ("mlp", 0, "mxfp8"),
    ],
)
@pytest.mark.parametrize(
    "rows,chain,first", [(4, 4, 1500), (16, 4, 900), (32, 4, 1500)]
)
def test_later_pass_parts_equal_their_stage_slices(
    rows, chain, first, experts, full_splits, part
):
    """A looped core's later pass launches a visit as its attention part
    (on the Q, K and V sources the regular path gated and normalized: a
    local layer's too) and its MoE part: each launch equals its stages run
    one launch a stage, writes only its own output, and gives every rank the
    same bits."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _part_worker,
        args=(
            port,
            ip.GEOMETRIES[1],
            rows,
            chain,
            first,
            experts,
            full_splits,
            part,
        ),
        nprocs=WORLD,
        join=True,
    )
