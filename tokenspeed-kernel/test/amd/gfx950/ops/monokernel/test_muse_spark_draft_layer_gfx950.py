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
import test_muse_spark_draft_attn_gfx950 as attn_t
import test_muse_spark_draft_mlp_gfx950 as mlp_t
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.benchmark import muse_spark_monokernel_gluon as gluon_inputs
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import payloads  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (  # noqa: E402
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    draft_attn,
    draft_layer,
    draft_mlp,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, OWNED, HEAD, HALF, Q_HEADS = 8, ref.HIDDEN, 1024, 128, 64, 8
SPLITS, PAGE, WINDOW_KEYS = attn_t.SPLITS, attn_t.PAGE, attn_t.WINDOW_KEYS


def plan(rows: int, last: bool) -> Plan:
    return draft_layer.layer_plan(rows, last)


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    """Within one BF16 step, or a step of the largest value where a sum
    cancels to near zero."""
    torch.testing.assert_close(
        got.float(),
        want.float(),
        rtol=2**-7,
        atol=want.float().abs().max().item() * 2**-9,
    )


def _worker(rank: int, port: int, prefix: int) -> None:
    rows = 8
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        # The last layer's plan is the larger; both share the scratch.
        p = plan(rows, True)
        scratch = ctx.zeros((p.scratch_pairs,), dtype=torch.int64)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        step = 31
        for layer in (2, 3, 4):
            host, weights, rt = gluon_inputs.draft_layer_inputs(
                prefix + layer, rank, rows, device
            )
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
                    peer_bases=group.bases,
                    layer=layer,
                    trace_buf=None,
                    compile_only=False,
                )
            torch.cuda.synchronize()
            _check(plan(rows, last), scratch, step, layer, host, rt, rank, rows)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _check(p, scratch, step, layer, host, rt, rank, rows) -> None:
    """Every stage of one launch on the kernel's own input."""
    own = slice(rank * OWNED, (rank + 1) * OWNED)
    tag = launch_tag(step, layer)
    copy = (step * 5 + layer) & 1
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
    _close(adl, (h.float() @ host["a_coef"].float().T).to(torch.bfloat16))
    a_delta = adl.view(rows, 2, ref.TAPS, ref.GROUPS)
    ao = bf("ao", Q_HEADS * HALF).view(rows, -1)
    attn_part = ref.grouped_conv(
        (ao.float() @ host["o_w"].float().T).to(torch.bfloat16),
        a_delta[:, 1],
        host["a_base"][1],
    )
    # The MLP half's input: every rank's partial, summed in rank order.
    gathered = [torch.empty_like(attn_part) for _ in range(WORLD)]
    dist.all_gather(gathered, attn_part)
    want_h, want_res = ref.tp_sum_norm(
        gathered, res1, host["post_gain"], round_tp_sum=True
    )
    _close(residual_out[:, own].cpu(), want_res[:, own])
    hin = bf("hin", HIDDEN // 2, copy=copy)
    _close(hin, want_h)
    # The MLP half's stages on the kernel's own input.
    mcp = words("cpart", 2 * ref.COEFFICIENTS).view(torch.float32)
    mdl = (mcp[:, : ref.COEFFICIENTS] + mcp[:, ref.COEFFICIENTS :]).to(torch.bfloat16)
    _close(mdl, (hin.float() @ host["m_coef"].float().T).to(torch.bfloat16))
    m_delta = mdl.view(rows, 2, ref.TAPS, ref.GROUPS)
    hc = bf("hc", HIDDEN // 2)
    _close(hc, ref.grouped_conv(hin, m_delta[:, 0], host["m_base"][0]))
    act = bf("act", ref.INTER_RANK // 2)
    gate, up = (
        (hc.float() @ host["gu_w"].float().T)
        .to(torch.bfloat16)
        .float()
        .split(ref.INTER_RANK, dim=-1)
    )
    _close(act, (torch.nn.functional.silu(gate) * up).to(torch.bfloat16))
    partial = (act.float() @ host["down_w"].float().T).to(torch.bfloat16)
    mlp_part = ref.grouped_conv(partial, m_delta[:, 1], host["m_base"][1])
    if rt["normed"] is None:
        _close(rt["out"].cpu(), mlp_part)
        return
    # The last layer: the final norm of the ranks' MLP partials and the
    # residual after the attention sublayer, the whole rows on every rank.
    parts = [torch.empty_like(mlp_part) for _ in range(WORLD)]
    dist.all_gather(parts, mlp_part)
    want_normed, _ = ref.tp_sum_norm(
        parts, want_res, host["final_w"], round_tp_sum=True
    )
    _close(rt["normed"].cpu(), want_normed)


@pytest.mark.parametrize("prefix", [3000])
def test_draft_layer_in_one_launch_hands_the_attention_partial_to_the_mlp(
    prefix: int,
):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, prefix), nprocs=WORLD, join=True)


def _two_requests_worker(rank: int, port: int, prefixes: tuple[int, int]) -> None:
    rows = 8
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        p = plan(2 * rows, True)
        scratch = ctx.zeros((p.scratch_pairs,), dtype=torch.int64)
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

            def run(rt, table, s):
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
                    peer_bases=group.bases,
                    layer=layer,
                    trace_buf=None,
                    compile_only=False,
                )
                torch.cuda.synchronize()
                dist.barrier()
                return outs["residual_out"], outs["normed" if last else "out"]

            one = [run(a, tables[0:1], step), run(b, tables[1:2], step + 1)]
            both = run(
                {
                    k: torch.cat([a[k], b[k]])
                    for k in ("x", "residual", "seq_len", "positions", "slots")
                },
                tables,
                step + 2,
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
    mp.spawn(_two_requests_worker, args=(port, prefixes), nprocs=WORLD, join=True)


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
