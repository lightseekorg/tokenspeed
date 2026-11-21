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

"""A whole DFlash2 draft layer (Muse Spark) in one launch across the 8 ranks."""

from __future__ import annotations

import contextlib
import socket
from datetime import timedelta

import dflash2_draft_reference as ref
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.benchmark import muse_spark_monokernel_gluon as gluon_inputs
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

import muse_spark_standins as standins  # noqa: E402
from monokernel_testing import payloads  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    draft_attn,
    draft_layer,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

WORLD, HIDDEN, OWNED, HEAD, HALF, Q_HEADS = 8, ref.HIDDEN, 1024, 128, 64, 8
PAGE = int(draft_attn.PAGE)
ALL = tuple(range(WORLD))


def plan(rows: int, last: bool) -> Plan:
    return draft_layer.layer_plan(rows, last)


def _copy(step: int, layer: int) -> int:
    """The peer copy of a launch: the copies alternate across forwards too."""
    return (step * int(draft_layer.DRAFT_LAYERS) + layer) & 1


def _inputs(prefix: int, rank: int, rows: int, device, present):
    """``gluon_inputs.draft_layer_inputs``; with ``present``, of a zero-shard
    model: the partials of the ranks outside it are zero, and every
    partial's, the residual's and the present ranks' o_proj and down rows'
    columns those ranks own are zero, so they put exactly zero (the input
    norm's, the MLP norm's and the final norm's red, ss and hin)."""
    host, weights, rt = gluon_inputs.draft_layer_inputs(prefix, rank, rows, device)
    if present is None:
        return host, weights, rt
    absent = standins.absent_columns(HIDDEN, OWNED, present)
    parts = [
        p if r in present else torch.zeros_like(p) for r, p in enumerate(host["parts"])
    ]
    for part in parts:
        part[:, absent] = 0
    host["parts"] = parts
    host["residual"][:, absent] = 0
    for name in ("o_w", "down_w"):
        host[name] = host[name].clone()
        host[name][absent] = 0  # the rows that write the absent columns
    weights = gluon_inputs.draft_layer_weights(host, rank, device)
    rt["x"] = parts[rank].to(device)
    rt["residual"] = host["residual"].to(device)
    return host, weights, rt


def _gather(t: torch.Tensor, present) -> list[torch.Tensor]:
    """Every rank's ``t`` in rank order: the running processes' gathered,
    zeros for the other ranks (the zero-shard model's)."""
    got = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(got, t)
    if present is None:
        return got
    whole = [torch.zeros_like(t) for _ in range(WORLD)]
    for r, part in zip(present, got):
        whole[r] = part
    return whole


def _worker(me: int, port: int, prefix: int, present) -> None:
    """``present``: None for the whole TP8 group, one process a rank; or the
    two rank programs that run (process ``me`` runs ``present[me]``) in a
    zero-shard model (:func:`_inputs`) whose other ranks' puts (exactly
    zero) the host writes before each launch."""
    rows = 8
    torch.cuda.set_device(me)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=me,
        world_size=WORLD if present is None else len(present),
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{me}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, me, device)
        # The last layer's plan is the larger; both share the scratch.
        p = plan(rows, True)
        scratch = ctx.zeros((p.scratch_pairs,), dtype=torch.int64)
        if present is None:
            rank, bases = me, group.bases
        else:
            rank = present[me]
            ghost = torch.zeros_like(scratch)
            bases = standins.peer_bases(group.bases, present, me, scratch, ghost)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        step = 31
        for layer in (2, 3, 4):
            host, weights, rt = _inputs(prefix + layer, rank, rows, device, present)
            last = layer == 4
            if last:
                # The last layer runs the draft's final norm in place of
                # storing its partial.
                g = torch.Generator().manual_seed(prefix)
                final_w = (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(
                    torch.bfloat16
                )
                host["final_w"] = final_w
                rt["out"] = None
                rt["final_gain"] = final_w[own].float().to(device)
                rt["normed"] = torch.zeros(
                    rows, HIDDEN, dtype=torch.bfloat16, device=device
                )
            if present is not None:
                standins.standin_fill(
                    scratch,
                    plan(rows, last),
                    present,
                    [(0, t) for t in range(rows)],
                    step,
                    layer,
                    _copy(step, layer),
                    True,
                )
            # The next layer, at another length, runs the first one's binary.
            no_compile = (
                assert_no_triton_compile(
                    draft_layer.gluon_muse_spark_draft_layer_gfx950
                )
                if layer == 3
                else contextlib.nullcontext()
            )
            with no_compile:
                draft_layer.launch_gluon_muse_spark_draft_layer_gfx950(
                    weights,
                    **rt,
                    scratch=scratch,
                    step=torch.full((1,), step, dtype=torch.int32, device=device),
                    peer_bases=bases,
                    layer=layer,
                    trace_buf=None,
                    compile_only=False,
                )
            torch.cuda.synchronize()
            _check(
                plan(rows, last), scratch, step, layer, host, rt, rank, rows, present
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _check(p, scratch, step, layer, host, rt, rank, rows, present) -> None:
    """Every stage of one launch on the kernel's own input. With
    ``present``, the absent ranks' partials are the zero-shard model's
    zeros, and their columns of the MLP input stay zero."""
    own = slice(rank * OWNED, (rank + 1) * OWNED)
    tag = launch_tag(step, layer)
    copy = _copy(step, layer)
    residual_out = rt["residual_out"]

    def words(name, width, copy=0):
        return torch.stack(
            [
                payloads(scratch, p, name, width, tag, row=r, copy=copy).cpu()
                for r in range(rows)
            ]
        )

    def bf(name, width, copy=0):
        return words(name, width, copy).view(torch.bfloat16)

    # The attention half up to its output (the half's own test covers
    # its stages one by one), then this rank's attention partial.
    _, res1 = ref.tp_sum_norm(
        host["parts"], host["residual"], host["in_gain"], round_tp_sum=True
    )
    h = bf("ahin", HIDDEN // 2, copy=copy)
    cpart = words("acpart", 2 * ref.COEFFICIENTS).view(torch.float32)
    adl = (cpart[:, : ref.COEFFICIENTS] + cpart[:, ref.COEFFICIENTS :]).to(
        torch.bfloat16
    )
    ref.assert_bf16_close(
        adl, (h.float() @ host["a_coef"].float().T).to(torch.bfloat16)
    )
    a_delta = adl.view(rows, 2, ref.TAPS, ref.GROUPS)
    ao = bf("ao", Q_HEADS * HALF).view(rows, -1)
    attn_part = ref.grouped_conv(
        (ao.float() @ host["o_w"].float().T).to(torch.bfloat16),
        a_delta[:, 1],
        host["a_base"][1],
    )
    # The MLP half's input: every rank's partial, summed in rank order.
    gathered = _gather(attn_part, present)
    want_h, want_res = ref.tp_sum_norm(
        gathered, res1, host["post_gain"], round_tp_sum=True
    )
    ref.assert_bf16_close(residual_out[:, own].cpu(), want_res[:, own])
    hin = bf("hin", HIDDEN // 2, copy=copy)
    ref.assert_bf16_close(hin, want_h)
    if present is not None:
        # The model is what the stand-ins claim: the absent ranks' columns
        # of the MLP input are zero, in both.
        absent = standins.absent_columns(HIDDEN, OWNED, present)
        assert not want_h[:, absent].float().any()
        assert not hin[:, absent].float().any()
    # The MLP half's stages on the kernel's own input.
    mcp = words("cpart", 2 * ref.COEFFICIENTS).view(torch.float32)
    mdl = (mcp[:, : ref.COEFFICIENTS] + mcp[:, ref.COEFFICIENTS :]).to(torch.bfloat16)
    ref.assert_bf16_close(
        mdl, (hin.float() @ host["m_coef"].float().T).to(torch.bfloat16)
    )
    m_delta = mdl.view(rows, 2, ref.TAPS, ref.GROUPS)
    hc = bf("hc", HIDDEN // 2)
    ref.assert_bf16_close(hc, ref.grouped_conv(hin, m_delta[:, 0], host["m_base"][0]))
    act = bf("act", ref.INTER_RANK // 2)
    gate, up = (
        (hc.float() @ host["gu_w"].float().T)
        .to(torch.bfloat16)
        .float()
        .split(ref.INTER_RANK, dim=-1)
    )
    ref.assert_bf16_close(act, (torch.nn.functional.silu(gate) * up).to(torch.bfloat16))
    partial = (act.float() @ host["down_w"].float().T).to(torch.bfloat16)
    mlp_part = ref.grouped_conv(partial, m_delta[:, 1], host["m_base"][1])
    if rt["normed"] is None:
        ref.assert_bf16_close(rt["out"].cpu(), mlp_part)
        return
    # The last layer: the final norm of the ranks' MLP partials and the
    # residual after the attention sublayer, the whole rows on every rank.
    parts = _gather(mlp_part, present)
    want_normed, _ = ref.tp_sum_norm(
        parts, want_res, host["final_w"], round_tp_sum=True
    )
    ref.assert_bf16_close(rt["normed"].cpu(), want_normed)


@pytest.mark.parametrize("prefix", [3000])
def test_draft_layer_in_one_launch_hands_the_attention_partial_to_the_mlp(
    prefix: int,
):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, prefix, None), nprocs=WORLD, join=True)


@pytest.mark.parametrize("present", standins.PAIRS, ids=standins.PAIR_IDS)
@pytest.mark.parametrize("prefix", [3000])
def test_draft_layer_in_one_launch_hands_the_attention_partial_to_the_mlp_on_two_gpus(
    prefix: int, present
):
    """As :func:`test_draft_layer_in_one_launch_hands_the_attention_partial_to_the_mlp`
    with the production rank programs ``present`` on two GPUs, in a
    zero-shard model (:func:`_inputs`) whose six other ranks put exactly
    zero (the host writes it): every stage against the reference on the
    kernel's own input, the last layer's final norm included, and the
    absent ranks' columns of the MLP input stay zero. Two non-zero terms
    per sum cannot check the rank order of the eight-rank all-reduces (see
    muse_spark_standins)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, prefix, present), nprocs=2, join=True)


def _two_requests_worker(
    me: int, port: int, prefixes: tuple[int, int], present
) -> None:
    """``present``: None for the whole TP8 group, one process a rank; or the
    two rank programs that run (process ``me`` runs ``present[me]``) while
    the host writes keyed stand-ins for the other ranks before each launch
    (:func:`muse_spark_standins.standin_fill`): keyed by request and row,
    so a launch of both requests and a launch of each read the same."""
    rows = 8
    torch.cuda.set_device(me)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=me,
        world_size=WORLD if present is None else len(present),
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{me}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, me, device)
        p = plan(2 * rows, True)
        scratch = ctx.zeros((p.scratch_pairs,), dtype=torch.int64)
        if present is None:
            rank, bases = me, group.bases
        else:
            rank = present[me]
            ghost = torch.zeros_like(scratch)
            bases = standins.peer_bases(group.bases, present, me, scratch, ghost)
        group.finish_prepare()
        g = torch.Generator().manual_seed(7)
        final_w = (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        step = 41
        for layer in (2, 4):
            last = layer == 4
            # One layer's weights; each request its own rows, context and
            # pages of one cache (request b's pages after request a's).
            _, weights, a = gluon_inputs.draft_layer_inputs(
                prefixes[0] + layer, rank, rows, device
            )
            _, _, b = gluon_inputs.draft_layer_inputs(
                prefixes[1] + layer, rank, rows, device
            )
            pages_a = a["k_cache"].shape[0] // PAGE
            b["page_table"] = b["page_table"] + pages_a
            b["slots"] = b["slots"] + pages_a * PAGE
            width = max(a["page_table"].shape[1], b["page_table"].shape[1])
            tables = torch.zeros(2, width, dtype=torch.int32, device=device)
            tables[0, : a["page_table"].shape[1]] = a["page_table"][0]
            tables[1, : b["page_table"].shape[1]] = b["page_table"][0]
            k_cache = torch.cat([a["k_cache"], b["k_cache"]])
            v_cache = torch.cat([a["v_cache"], b["v_cache"]])
            cos_sin = max(a["cos_sin"], b["cos_sin"], key=len)

            def run(rt, table, s, keys):
                if present is not None:
                    standins.standin_fill(
                        scratch,
                        plan(len(keys), last),
                        present,
                        keys,
                        s,
                        layer,
                        _copy(s, layer),
                        False,
                    )
                outs = dict(
                    residual_out=torch.zeros_like(rt["residual"]),
                    out=None if last else torch.zeros_like(rt["x"]),
                    final_gain=final_w[own].float().to(device) if last else None,
                    normed=torch.zeros_like(rt["x"]) if last else None,
                )
                draft_layer.launch_gluon_muse_spark_draft_layer_gfx950(
                    weights,
                    x=rt["x"],
                    residual=rt["residual"],
                    seq_len=rt["seq_len"],
                    positions=rt["positions"],
                    slots=rt["slots"],
                    page_table=table,
                    # Each launch its own copy: the launches write the
                    # block's keys into the cache.
                    k_cache=k_cache.clone(),
                    v_cache=v_cache.clone(),
                    cos_sin=cos_sin,
                    **outs,
                    scratch=scratch,
                    step=torch.full((1,), s, dtype=torch.int32, device=device),
                    peer_bases=bases,
                    layer=layer,
                    trace_buf=None,
                    compile_only=False,
                )
                torch.cuda.synchronize()
                dist.barrier()
                return outs["residual_out"], outs["normed" if last else "out"]

            keys_a = [(0, t) for t in range(rows)]
            keys_b = [(1, t) for t in range(rows)]
            one = [
                run(a, tables[0:1], step, keys_a),
                run(b, tables[1:2], step + 1, keys_b),
            ]
            both = run(
                {
                    k: torch.cat([a[k], b[k]])
                    for k in ("x", "residual", "seq_len", "positions", "slots")
                },
                tables,
                step + 2,
                keys_a + keys_b,
            )
            step += 3
            for q in range(2):
                block = slice(q * rows, (q + 1) * rows)
                # The residual: this rank's columns (the others are not
                # stored); the output: every column.
                assert torch.equal(both[0][block, own], one[q][0][:, own])
                assert torch.equal(both[1][block], one[q][1]), (layer, q)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("prefixes", [(3000, 9000)])
def test_draft_layer_of_two_requests_matches_a_launch_of_each(
    prefixes: tuple[int, int],
):
    """One launch of two requests' blocks gives each request the bits of a
    launch of its block alone: the rows are independent except attention,
    which runs per request (its own length, pages and keys)."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_two_requests_worker, args=(port, prefixes, None), nprocs=WORLD, join=True)


@pytest.mark.parametrize("present", standins.PAIRS, ids=standins.PAIR_IDS)
@pytest.mark.parametrize("prefixes", [(3000, 9000)])
def test_draft_layer_of_two_requests_matches_a_launch_of_each_on_two_gpus(
    prefixes: tuple[int, int], present
):
    """As :func:`test_draft_layer_of_two_requests_matches_a_launch_of_each`
    with the production rank programs ``present`` on two GPUs: the other
    ranks' puts are keyed stand-ins (by request and row), so each request's
    bits from the launch of both match its own launch's exactly. Two
    processes cannot check the values the present ranks put for the absent
    ones (see muse_spark_standins)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_two_requests_worker, args=(port, prefixes, present), nprocs=2, join=True)


def test_one_binary_serves_every_layer_step_and_length():
    """Each launch changes the layer, the step, the length, the rows'
    positions and slots and the page table's width: compiling the launches
    for one GPU's rank finds the first launch's binary. The last layer (it
    runs the final norm) is another plan, which the guard must see
    compile."""
    rows, device = 8, torch.device("cuda:0")
    # Compiling alone reads none of these: plain device memory stands in for
    # the symmetric heap and the peers.
    scratch = torch.zeros(
        plan(rows, True).scratch_pairs, dtype=torch.int64, device=device
    )
    peer_bases = torch.zeros(WORLD, dtype=torch.int64, device=device)

    def compile_launch(layer: int, step: int, prefix: int) -> None:
        _, weights, rt = gluon_inputs.draft_layer_inputs(prefix, 0, rows, device)
        if layer == 4:
            rt["out"] = None
            rt["final_gain"] = torch.ones(OWNED, device=device)
            rt["normed"] = torch.zeros(
                rows, HIDDEN, dtype=torch.bfloat16, device=device
            )
        draft_layer.launch_gluon_muse_spark_draft_layer_gfx950(
            weights,
            **rt,
            scratch=scratch,
            step=torch.full((1,), step, dtype=torch.int32, device=device),
            peer_bases=peer_bases,
            layer=layer,
            trace_buf=None,
            compile_only=True,
        )

    compile_launch(2, 31, 3000)
    with assert_no_triton_compile(draft_layer.gluon_muse_spark_draft_layer_gfx950):
        for layer, step, prefix in ((3, 32, 3001), (2, 33, 900), (3, 34, 9000)):
            compile_launch(layer, step, prefix)
    # The guard is live: the last layer's plan compiles.
    with pytest.raises(AssertionError, match="compiled 1 new"):
        with assert_no_triton_compile(draft_layer.gluon_muse_spark_draft_layer_gfx950):
            compile_launch(4, 35, 3000)
