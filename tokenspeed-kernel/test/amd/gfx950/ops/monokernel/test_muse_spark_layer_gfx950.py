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

"""A whole Muse Spark decode layer as one persistent launch per TP8 rank,
for one decode row or a speculative verify's chained rows."""

from __future__ import annotations

import dataclasses
import os
import socket
import subprocess
import sys
import textwrap
from datetime import timedelta

import muse_spark_reference as ref
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import payloads  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel import stuck  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    layer,
    moe_tail,
    stuck_report,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.qkvg import (  # noqa: E402
    PROJ_WORDS,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.weights import (  # noqa: E402
    pack_layer,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import launch_tag  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, POOLED, OWNED, EXPERTS = 8, 8192, 4096, 1024, 16
QKVG, Q = 2304, 1024


def _shared(seed: int, length: int, rows: int):
    """Inputs every rank shares: ``a_in`` and ``h`` of each row (``[rows,
    8192]``; row ``t`` at position ``length - 1 + t``), the weights of the
    norms and the router, and the page table."""
    g = torch.Generator().manual_seed(seed)

    def gain(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16).float()

    a_in = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
    h = torch.randn(rows, HIDDEN, generator=g) * 2
    gates = [torch.rand(HIDDEN, generator=g) + 0.5 for _ in range(4)]
    router = torch.randn(256, HIDDEN, generator=g) / HIDDEN**0.5
    bias = torch.randn(256, generator=g) * 0.1
    bias[EXPERTS:] = -1e4  # the test holds 16 experts per rank
    pages = (length + rows - 1 + 63) // 64
    table = (torch.randperm(pages + 3, generator=g)[:pages] + 1).to(torch.int32)
    return dict(
        a_in=a_in,
        h=h,
        alpha1=gates[0],
        beta1=gates[1],
        alpha2=gates[2],
        beta2=gates[3],
        pre_gain=gain(HIDDEN),
        pre_exp_gain=gain(POOLED),
        pen_gain=gain(POOLED),
        post_gain=gain(HIDDEN),
        in_gain=gain(HIDDEN),
        router=router,
        bias=bias,
        table=table,
    )


def _rank_weights(rank: int, length: int):
    g = torch.Generator().manual_seed(1000 + rank)

    def bf(*shape, k):
        return (torch.randn(*shape, generator=g) / k**0.5).to(torch.bfloat16)

    slots = ((length + 63) // 64 + 5) * 64
    return dict(
        w_qkvg=bf(QKVG, HIDDEN, k=HIDDEN),
        w_o=bf(HIDDEN, Q, k=Q),
        w_pool=bf(512, HIDDEN, k=HIDDEN),
        w_post=bf(OWNED, POOLED, k=POOLED),
        k_cache=torch.randn(slots, 2, 64, generator=g).to(torch.bfloat16),
        v_cache=torch.randn(slots, 2, 64, generator=g).to(torch.bfloat16),
    )


def _all_experts(t: torch.Tensor) -> torch.Tensor:
    """``t`` (the test's 16 experts) in a tensor of all 256 experts with the
    same strides, as the packer requires: the bias routes only to the first
    16, so the others are never read."""
    if t.stride(0) != max(t.stride()):
        raise AssertionError("experts are not the outer dimension")
    full = torch.empty_strided(
        (256, *t.shape[1:]), t.stride(), dtype=t.dtype, device=t.device
    )
    full.zero_()
    full[: t.shape[0]].copy_(t)
    return full


def _reference(sh, length, full, kernel_pooled, kernel_hp, dev):
    """Each row's pooled row, FP32 residual ``h''`` and next BF16 attention
    input (``[rows, ...]``).

    MXFP4 quantization is discontinuous: a pooled value one BF16 step from
    the reference's can quantize one E2M1 step away and move every expert
    output a little. So the pooled rows are checked on their own, and the
    expert math downstream of them starts from the kernel's
    ``kernel_pooled``. In the same way the FFN residual gate starts from the
    kernel's ``h'`` (``kernel_hp``), which is checked on its own.
    """
    rows = sh["a_in"].shape[0]
    end = length + rows - 1
    window = 0 if full else layer.WINDOW
    splits = layer.FULL_SPLITS if full else layer.SLIDING_SPLITS
    cs = ref.rope_table(end + 1)
    b = torch.zeros(rows, HIDDEN)
    for r in range(WORLD):
        w = _rank_weights(r, end)
        qkvg = (sh["a_in"].float() @ w["w_qkvg"].float().T).to(torch.bfloat16)
        z, _, _, _ = ref.attention_z(
            qkvg, w["k_cache"], w["v_cache"], sh["table"], length, window, splits, cs
        )
        b = b + (z.float() @ w["w_o"].float().T).to(torch.bfloat16).float()
    b = b.to(torch.bfloat16).float()
    b = b * torch.rsqrt(b.pow(2).mean(-1, keepdim=True) + 1e-8)
    hp = sh["beta1"] * b + sh["alpha1"] * sh["h"]
    u = hp * torch.rsqrt(hp.pow(2).mean(-1, keepdim=True) + 1e-5) * sh["pre_gain"]
    w_pool = torch.cat([_rank_weights(r, end)["w_pool"] for r in range(WORLD)])
    pooled = (u.to(torch.bfloat16).float() @ w_pool.float().T).to(torch.bfloat16)
    w_post = torch.cat([_rank_weights(r, end)["w_post"] for r in range(WORLD)])
    h2 = torch.empty(rows, HIDDEN)
    a = torch.empty(rows, HIDDEN, dtype=torch.bfloat16)
    gated = torch.empty(rows, HIDDEN)
    score = torch.sigmoid((u.double() @ sh["router"].double().T) * 0.17677669529663687)
    ids = torch.topk(score + sh["bias"].double(), 8).indices  # [rows, 8]
    # The experts on the worker's GPU, every row at once.
    routes = torch.zeros(rows, 8, POOLED)
    for r in range(WORLD):
        routes += (
            ref.expert_partials(
                r, EXPERTS, kernel_pooled.to(dev), sh["pre_exp_gain"], ids.to(dev)
            )
            .float()
            .cpu()
        )
    for t in range(rows):
        weights = (score[t, ids[t]] / score[t, ids[t]].sum()).float()
        p = routes[t].to(torch.bfloat16).float() * sh["pen_gain"]
        p = p * torch.rsqrt(p.pow(2).mean(1, keepdim=True) + 1e-8)
        combined = (p * weights[:, None]).sum(0).to(torch.bfloat16)
        y = (combined.float() @ w_post.float().T).to(torch.bfloat16).float()
        y = y * sh["post_gain"]
        y = y * torch.rsqrt(y.pow(2).mean() + 1e-8)
        h2[t] = sh["beta2"] * y + sh["alpha2"] * kernel_hp[t]
        a[t] = (h2[t] * torch.rsqrt(h2[t].pow(2).mean() + 1e-5) * sh["in_gain"]).to(
            torch.bfloat16
        )
        gated[t] = (sh["beta2"] * y).abs()
    return pooled, hp, (sh["beta1"] * b).abs(), h2, a, gated


def _pack(
    rank: int,
    dev: torch.device,
    sh: dict,
    length: int,
    full: bool,
):
    """Pack ``rank``'s layer from the regular path's tensors (``length``: the
    last row's length)."""
    return pack_layer(**_layer_tensors(rank, dev, sh, length, full))


def _layer_tensors(
    rank: int,
    dev: torch.device,
    sh: dict,
    length: int,
    full: bool,
) -> dict:
    """The packer's tensors by role (:func:`pack_layer`'s arguments, which the
    kernel boundary's ``prepare_muse_spark_layer`` takes as ``tensors``)."""
    w = _rank_weights(rank, length)
    w13, w13s, w2, w2s = ref.preprocessed_experts(rank, EXPERTS, dev)
    experts = tuple(_all_experts(t) for t in (w13, w13s, w2, w2s))
    # The packer takes the regular path's tensors: the replicated pooler
    # and post-expert weights, and the router as the regular path stores
    # it (logits are x @ router); it slices this rank's part.
    return dict(
        rank=rank,
        full=full,
        qkvg=w["w_qkvg"].to(dev),
        o=w["w_o"].to(dev),
        alpha1=sh["alpha1"].to(dev),
        beta1=sh["beta1"].to(dev),
        pre_gain=sh["pre_gain"].to(dev),
        router=sh["router"].t().to(dev),
        router_bias=sh["bias"].to(dev),
        pool=torch.cat([_rank_weights(r, length)["w_pool"] for r in range(WORLD)]).to(
            dev
        ),
        experts=experts,
        pre_exp_gain=sh["pre_exp_gain"].to(dev),
        pen_gain=sh["pen_gain"].to(dev),
        post=torch.cat([_rank_weights(r, length)["w_post"] for r in range(WORLD)]).to(
            dev
        ),
        post_gain=sh["post_gain"].to(dev),
        alpha2=sh["alpha2"].to(dev),
        beta2=sh["beta2"].to(dev),
        in_gain=sh["in_gain"].to(dev),
    )


def _worker(rank: int, port: int, length: int, full: bool, rows: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=900),
    )
    try:
        dev = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
        plan = layer.layer_plan(
            full=full,
            rows=rows,
            chain=rows,
            warm_next=True,
            own_qkvg=True,
            next_qkvg=False,
            check_parity=False,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        # A draft's taps: the launch stores its residual into slot 3 of
        # every rank's copy.
        tap = ctx.zeros((rows, 5, HIDDEN), dtype=torch.bfloat16)
        end = length + rows - 1
        sh = _shared(42, length, rows)
        w = _rank_weights(rank, end)
        weights = _pack(rank, dev, sh, end, full)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        k_cache, v_cache = w["k_cache"].to(dev), w["v_cache"].to(dev)
        h_in = sh["h"][:, own].contiguous().to(dev)
        a_in = sh["a_in"].to(dev)
        h_next = torch.empty(rows, OWNED, device=dev)
        h_whole = torch.empty(rows, HIDDEN, device=dev)
        a_out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)

        def launch(
            index: int, step: int, seq_len: int, compile_only: bool, whole_rows=False
        ) -> None:
            pos = torch.arange(seq_len - 1, seq_len - 1 + rows)
            slots = sh["table"][pos // 64].long() * 64 + pos % 64
            layer.launch_gluon_muse_spark_layer_gfx950(
                weights,
                h=h_in,
                a=a_in,
                h_out=h_whole if whole_rows else h_next,
                a_out=a_out,
                seq_len=(pos + 1).to(torch.int32).to(dev),
                slot=slots.to(torch.int32).to(dev),
                page_table=sh["table"].to(dev),
                k_cache=k_cache,
                v_cache=v_cache,
                cos_sin=ref.rope_table(end + 1).to(dev),
                scratch=scratch,
                step=torch.full((1,), step, dtype=torch.int32, device=dev),
                peer_bases=group.bases,
                layer=index,
                chain=rows,
                whole_rows=whole_rows,
                tap=tap,
                tap_slot=3,
                # The warm stage reads weights only: it runs here (on the
                # layer's own QKVG weights) and changes no output.
                warm_next=weights,
                proj_in=None,
                proj_out=None,
                trace_buf=None,
                stuck_rec=None,
                # Launches repeat a layer between barriers, which the check
                # cannot tell from overlapping launches (the qnext chain
                # test checks).
                check_parity=False,
                compile_only=compile_only,
            )

        group.finish_prepare()
        lay = 1 if full else 2
        # Compiling alone runs nothing: the outputs and the scratch keep
        # their fill (until every rank has checked: a peer's launch puts
        # into this scratch). The launch then runs the binary it compiled.
        h_next.fill_(float("nan"))
        launch(lay, 3, length, compile_only=True)
        torch.cuda.synchronize()
        assert torch.isnan(h_next).all()
        assert not scratch.any()
        dist.barrier()
        with assert_no_triton_compile(layer.gluon_muse_spark_layer_gfx950):
            launch(lay, 3, length, compile_only=False)
            torch.cuda.synchronize()
        dist.barrier()
        tag = launch_tag(3, lay)
        kernel_pooled = torch.stack(
            [
                payloads(scratch, plan, "cg", POOLED // 2, tag, row=t, copy=lay & 1)
                .view(torch.bfloat16)
                .cpu()
                for t in range(rows)
            ]
        )
        hp_own = torch.stack(
            [
                payloads(scratch, plan, "hp", OWNED, tag, row=t)
                .view(torch.float32)
                .cpu()
                for t in range(rows)
            ]
        )
        slices = [torch.empty_like(hp_own) for _ in range(WORLD)]
        dist.all_gather(slices, hp_own)
        kernel_hp = torch.cat(slices, dim=1)
        pooled, hp, gated1, h2, a, gated = _reference(
            sh, length, full, kernel_pooled, kernel_hp, dev
        )
        # h' sums 8 ranks' BF16 partials: where they cancel, a step of one
        # partial is an absolute error at the row's scale, which the norm
        # carries into beta b.
        err1 = (hp_own - hp[:, own]).abs()
        bound1 = 2**-7 * hp[:, own].abs() + 2**-6 * gated1.amax(-1, keepdim=True)
        assert torch.all(err1 <= bound1), float(err1.max())
        # The pooled row is the BF16 projection of bf16(u), which carries the
        # attention tail's BF16 steps: up to two steps, and near zero an
        # absolute error of the row's scale (the sum cancels).
        for t in range(rows):
            atol = 2**-9 * float(pooled[t].float().abs().max())
            torch.testing.assert_close(
                kernel_pooled[t].float(), pooled[t].float(), rtol=2**-6, atol=atol
            )
        # The tap: every rank's h'' in its columns, rounded once to BF16, the
        # other slots untouched.
        parts = [torch.empty_like(h_next.cpu()) for _ in range(WORLD)]
        dist.all_gather(parts, h_next.cpu())
        torch.cuda.synchronize()
        dist.barrier()
        got_tap = tap.cpu()
        assert torch.equal(got_tap[:, 3], torch.cat(parts, dim=1).to(torch.bfloat16))
        assert not got_tap[:, [0, 1, 2, 4]].any()
        err = (h_next.cpu() - h2[:, own]).abs()
        # In the same way, a BF16 step of the projection is an absolute
        # error at the scale of beta y.
        bound = 2**-7 * h2[:, own].abs() + 2**-6 * gated.amax(-1, keepdim=True)
        assert torch.all(err <= bound), float(err.max())
        # The next input is BF16 of h'' after a norm: the same row-scale error.
        for t in range(rows):
            atol = 2**-8 * float(a[t].float().abs().max())
            torch.testing.assert_close(
                a_out[t].float().cpu(), a[t].float(), rtol=2**-6, atol=atol
            )
        # Before a regular layer, every rank stores whole residual rows: the
        # gather of the owned columns, bit for bit, and the same next input.
        owned_a_out = a_out.cpu()
        launch(lay, 4, length, compile_only=True, whole_rows=True)
        torch.cuda.synchronize()
        dist.barrier()
        launch(lay, 4, length, compile_only=False, whole_rows=True)
        torch.cuda.synchronize()
        dist.barrier()
        assert torch.equal(h_whole.cpu(), torch.cat(parts, dim=1))
        assert torch.equal(a_out.cpu(), owned_a_out)
        dist.barrier()
        # Serving moves the layer index, the step and the length every
        # launch; they are runtime arguments: one binary serves them all.
        # Each step flips the peer copy (layer & 1), as serving does.
        with assert_no_triton_compile(layer.gluon_muse_spark_layer_gfx950):
            for step, (index, seq_len) in enumerate(
                ((lay + 1, length), (lay + 18, length - 1), (lay + 35, length - 70)),
                start=4,
            ):
                launch(index, step, seq_len, compile_only=False)
            torch.cuda.synchronize()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_the_kernel_boundary_refuses_experts_of_another_layout():
    """Experts that a regular MoE kernel with another weight layout prepared
    (here the Triton MXFP4 one) are not the layer kernel's: packing them
    raises instead of reading the wrong layout."""
    from tokenspeed_kernel.ops.model.muse_spark.monokernel import (
        prepare_muse_spark_layer,
    )

    dev = torch.device("cuda:0")
    tensors = _layer_tensors(0, dev, _shared(42, 3000, 1), 3000, False)
    tensors["experts_kernel"] = "triton_mxfp4_precomputed_moe_apply"
    with pytest.raises(ValueError, match="triton_mxfp4_precomputed_moe_apply"):
        prepare_muse_spark_layer(tensors=tensors, max_rows=layer.MAX_ROWS)


@pytest.mark.parametrize("full", [False, True])
def test_the_kernel_boundary_packs_a_layer_for_its_launches(full):
    """The runtime packs a layer through the kernel boundary
    (``prepare_muse_spark_layer``): it returns the packed layer and the
    mailbox scratch that the plans of every row count up to the launch's
    rows need."""
    from tokenspeed_kernel.ops.model.muse_spark.monokernel import (
        FusedDecodeLayer,
        prepare_muse_spark_layer,
    )

    dev = torch.device("cuda:0")
    tensors = _layer_tensors(0, dev, _shared(42, 3000, 1), 3000, full)
    tensors["experts_kernel"] = "gluon_mxfp4_precomputed_moe_apply"
    fused = prepare_muse_spark_layer(tensors=tensors, max_rows=layer.MAX_ROWS)

    assert isinstance(fused, FusedDecodeLayer)
    assert fused.weights.full is full and fused.weights.rank == 0
    widest = layer.layer_plan(
        full=full,
        rows=layer.MAX_ROWS,
        chain=layer.MAX_ROWS,
        warm_next=True,
        own_qkvg=True,
        next_qkvg=False,
        check_parity=False,
    )
    assert fused.scratch_pairs >= widest.scratch_pairs


@pytest.mark.parametrize("full", [False, True])
def test_one_binary_serves_every_layer_step_and_length(full):
    """What serving changes from launch to launch (the layer index, the step,
    the lengths and the KV slots) reaches the kernel at runtime: compiling
    the launches for one GPU's rank finds the first launch's binary. A new
    row count is a new plan, which the guard must see compile."""
    rows, length = 8, 3000
    end = length + rows - 1
    dev = torch.device("cuda:0")
    sh = _shared(42, length, rows)
    w = _rank_weights(0, end)
    weights = _pack(0, dev, sh, end, full)
    k_cache, v_cache = w["k_cache"].to(dev), w["v_cache"].to(dev)
    # Compiling alone reads none of these: plain device memory stands in for
    # the symmetric heap and the peers.
    widest = layer.layer_plan(
        full=full,
        rows=rows,
        chain=rows,
        warm_next=True,
        own_qkvg=True,
        next_qkvg=False,
        check_parity=False,
    )
    scratch = torch.zeros(widest.scratch_pairs, dtype=torch.int64, device=dev)
    peer_bases = torch.zeros(WORLD, dtype=torch.int64, device=dev)

    def compile_launch(index: int, step: int, seq_len: int, rows: int) -> None:
        pos = torch.arange(seq_len - 1, seq_len - 1 + rows)
        slots = sh["table"][pos // 64].long() * 64 + pos % 64
        layer.launch_gluon_muse_spark_layer_gfx950(
            weights,
            h=torch.zeros(rows, OWNED, device=dev),
            a=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=dev),
            h_out=torch.empty(rows, OWNED, device=dev),
            a_out=torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev),
            seq_len=(pos + 1).to(torch.int32).to(dev),
            slot=slots.to(torch.int32).to(dev),
            page_table=sh["table"].to(dev),
            k_cache=k_cache,
            v_cache=v_cache,
            cos_sin=ref.rope_table(end + 1).to(dev),
            scratch=scratch,
            step=torch.full((1,), step, dtype=torch.int32, device=dev),
            peer_bases=peer_bases,
            layer=index,
            chain=rows,
            whole_rows=False,
            tap=torch.zeros(rows, 5, HIDDEN, dtype=torch.bfloat16, device=dev),
            tap_slot=3,
            warm_next=weights,
            proj_in=None,
            proj_out=None,
            trace_buf=None,
            stuck_rec=None,
            check_parity=False,
            compile_only=True,
        )

    lay = 1 if full else 2
    compile_launch(lay, 3, length, rows)
    with assert_no_triton_compile(layer.gluon_muse_spark_layer_gfx950):
        for step, (index, seq_len) in enumerate(
            ((lay + 1, length), (lay + 18, length - 1), (lay + 35, length - 70)),
            start=4,
        ):
            compile_launch(index, step, seq_len, rows)
    # The guard is live: a row count the server did not warm compiles.
    with pytest.raises(AssertionError, match="compiled 1 new"):
        with assert_no_triton_compile(layer.gluon_muse_spark_layer_gfx950):
            compile_launch(lay, 7, length, rows - 1)


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("full", [False, True])
def test_layer_matches_the_regular_path_math(full, rows):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, 3000, full, rows), nprocs=WORLD, join=True)


def _two_requests_worker(rank: int, port: int, full: bool) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=900),
    )
    try:
        dev = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
        rows = 8  # a verify of 7 draft tokens a request
        plan = layer.layer_plan(
            full=full,
            rows=2 * rows,
            chain=rows,
            warm_next=False,
            own_qkvg=True,
            next_qkvg=False,
            check_parity=False,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        lengths = (3000, 3100)
        ends = [n + rows - 1 for n in lengths]
        # Request 0 is _shared's; request 1 has inputs and pages of its own.
        sh = _shared(42, lengths[0], rows)
        g = torch.Generator().manual_seed(7)
        reqs = [
            dict(a=sh["a_in"], h=sh["h"], table=sh["table"]),
            dict(
                a=torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16),
                h=torch.randn(rows, HIDDEN, generator=g) * 2,
                table=(
                    int(sh["table"].max())
                    + 1
                    + torch.randperm((ends[1] + 63) // 64, generator=g)
                ).to(torch.int32),
            ),
        ]
        w = _rank_weights(rank, 3 * max(ends))
        weights = _pack(rank, dev, sh, ends[1], full)
        cos_sin = ref.rope_table(max(ends) + 1).to(dev)
        for b, r in enumerate(reqs):
            pos = torch.arange(lengths[b] - 1, lengths[b] - 1 + rows)
            r["seq_len"] = (pos + 1).to(torch.int32)
            r["slot"] = (r["table"][pos // 64].long() * 64 + pos % 64).to(torch.int32)

        def launch(rs, step, k_cache, v_cache):
            n = len(rs) * rows
            h_out = torch.empty(n, OWNED, device=dev)
            a_out = torch.empty(n, HIDDEN, dtype=torch.bfloat16, device=dev)
            width = max(len(r["table"]) for r in rs)
            table = torch.zeros(len(rs), width, dtype=torch.int32)
            for b, r in enumerate(rs):
                table[b, : len(r["table"])] = r["table"]
            layer.launch_gluon_muse_spark_layer_gfx950(
                weights,
                h=torch.cat([r["h"][:, own] for r in rs]).contiguous().to(dev),
                a=torch.cat([r["a"] for r in rs]).to(dev),
                h_out=h_out,
                a_out=a_out,
                seq_len=torch.cat([r["seq_len"] for r in rs]).to(dev),
                slot=torch.cat([r["slot"] for r in rs]).to(dev),
                page_table=table.to(dev),
                k_cache=k_cache,
                v_cache=v_cache,
                cos_sin=cos_sin,
                scratch=scratch,
                step=torch.full((1,), step, dtype=torch.int32, device=dev),
                peer_bases=group.bases,
                layer=1 if full else 2,
                chain=rows,
                whole_rows=False,
                tap=None,
                tap_slot=None,
                warm_next=None,
                proj_in=None,
                proj_out=None,
                trace_buf=None,
                stuck_rec=None,
                # Repeats one layer between barriers (see _worker).
                check_parity=False,
                compile_only=False,
            )
            torch.cuda.synchronize()
            dist.barrier()
            return h_out.cpu(), a_out.cpu()

        group.finish_prepare()
        caches = [w["k_cache"].to(dev), w["v_cache"].to(dev)]
        apart = [c.clone() for c in caches]
        one = [launch([r], 3 + b, *apart) for b, r in enumerate(reqs)]
        together = [c.clone() for c in caches]
        h2, a2 = launch(reqs, 5, *together)
        # Every row's arithmetic is its own: the same values as each
        # request in a launch of its own. A row's FP32 sums reduce across
        # lanes in a tree that the launch's row count shapes, so the FP32
        # residual may differ by a rounding step of a row factor (an
        # absolute error at the row's scale), and the next input, its BF16
        # norm, may then round a value the other way; the caches are the
        # same bits.
        want = torch.cat([o[0] for o in one])
        scale = want.abs().amax(dim=1, keepdim=True)
        want_a = torch.cat([o[1] for o in one]).float()
        assert torch.all((h2 - want).abs() <= 2**-21 * scale)
        assert torch.all((a2.float() - want_a).abs() <= 2**-7 * want_a.abs())
        assert torch.equal(together[0], apart[0])
        assert torch.equal(together[1], apart[1])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("full", [False, True])
def test_two_requests_in_one_launch_match_their_own_launches(full):
    """The verify rows of two requests (their own lengths and page tables)
    in one launch give what each request's own launch gives."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_two_requests_worker, args=(port, full), nprocs=WORLD, join=True)


def _shift_worker(rank: int, port: int, kinds: tuple[bool, ...]) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=900),
    )
    try:
        dev = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
        rows, length = 8, 3000
        end = length + rows - 1
        plan = layer.layer_plan(
            full=False,
            rows=rows,
            chain=rows,
            warm_next=False,
            own_qkvg=True,
            next_qkvg=False,
            check_parity=True,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        sh = _shared(42, length, rows)
        w = _rank_weights(rank, end)
        packed = _pack(rank, dev, sh, end, kinds[0])
        # Each layer its own kind and q | k | v | gate weights: a launch
        # must project with the next layer's weights, not its own.
        layers = []
        for i, full in enumerate(kinds):
            g = torch.Generator().manual_seed(2000 + 10 * rank + i)
            qkvg = (torch.randn(QKVG, HIDDEN, generator=g) / HIDDEN**0.5).to(
                torch.bfloat16
            )
            layers.append(
                dataclasses.replace(packed, full=full, qkvg=bf16_tiles(qkvg.to(dev)))
            )
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        pos = torch.arange(length - 1, length - 1 + rows)
        slots = (sh["table"][pos // 64].long() * 64 + pos % 64).to(torch.int32)
        cos_sin = ref.rope_table(end + 1).to(dev)
        proj = torch.zeros(2, rows, int(PROJ_WORDS), dtype=torch.int32, device=dev)

        # Layer index ``index`` (the tag and the peer copy) moves on every
        # launch, as in serving: the second pass goes on from the first.
        def launch(i, index, h, a, step, caches, proj_in, proj_out):
            h_out = torch.empty(rows, OWNED, device=dev)
            a_out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
            layer.launch_gluon_muse_spark_layer_gfx950(
                layers[i],
                h=h,
                a=a,
                h_out=h_out,
                a_out=a_out,
                seq_len=(pos + 1).to(torch.int32).to(dev),
                slot=slots.to(dev),
                page_table=sh["table"].to(dev),
                k_cache=caches[0],
                v_cache=caches[1],
                cos_sin=cos_sin,
                scratch=scratch,
                step=torch.full((1,), step, dtype=torch.int32, device=dev),
                peer_bases=group.bases,
                layer=index,
                chain=rows,
                whole_rows=False,
                tap=None,
                tap_slot=None,
                warm_next=None if proj_out is None else layers[i + 1],
                proj_in=proj_in,
                proj_out=proj_out,
                trace_buf=None,
                stuck_rec=None,
                check_parity=True,
                compile_only=False,
            )
            torch.cuda.synchronize()
            dist.barrier()
            return h_out, a_out

        group.finish_prepare()
        n = len(kinds)
        caches = [[w["k_cache"].to(dev), w["v_cache"].to(dev)] for _ in range(n)]
        regular = [[c.clone() for c in cs] for cs in caches]
        shifted = [[c.clone() for c in cs] for cs in caches]
        h, a = sh["h"][:, own].contiguous().to(dev), sh["a_in"].to(dev)
        want = []
        for i in range(n):
            h, a = launch(i, i + 1, h, a, 3, regular[i], None, None)
            want.append((h.cpu(), a.cpu()))
        # The first launch projects its own q | k | v and the next layer's;
        # the others read theirs from the previous launch; the last projects
        # no next one.
        h, a = sh["h"][:, own].contiguous().to(dev), sh["a_in"].to(dev)
        for i in range(n):
            h, a = launch(
                i,
                n + i + 1,
                h,
                a,
                4,
                shifted[i],
                None if i == 0 else proj[i % 2],
                None if i == n - 1 else proj[(i + 1) % 2],
            )
            assert torch.equal(h.cpu(), want[i][0]), f"layer {i}: residual"
            assert torch.equal(a.cpu(), want[i][1]), f"layer {i}: next input"
            assert torch.equal(shifted[i][0], regular[i][0]), f"layer {i}: k cache"
            assert torch.equal(shifted[i][1], regular[i][1]), f"layer {i}: v cache"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kinds", [(True, False, False), (False, False, True)])
def test_launches_that_project_the_next_layer_match_the_regular_order(kinds):
    """A launch that ends with the next layer's q | k | v projection (qnext),
    and a launch that reads it instead of projecting its own, give the same
    bits as launches that each project their own: a full layer before two
    sliding ones, and a sliding one between a sliding and a full one."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_shift_worker, args=(port, kinds), nprocs=WORLD, join=True)


def _graph_worker(rank: int, port: int, length: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=900),
    )
    try:
        dev = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
        plan = layer.layer_plan(
            full=False,
            rows=1,
            chain=1,
            warm_next=False,
            own_qkvg=True,
            next_qkvg=False,
            check_parity=False,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        sh = _shared(42, length, 1)
        weights = _pack(rank, dev, sh, length, False)
        w = _rank_weights(rank, length)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        k_cache, v_cache = w["k_cache"].to(dev), w["v_cache"].to(dev)
        h_in = sh["h"][:, own].contiguous().to(dev)
        a_in = sh["a_in"].to(dev)
        h_out = torch.empty(1, OWNED, device=dev)
        a_out = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        # The per-step operands are device tensors that the decode metadata
        # refreshes in place, as in serving.
        step = torch.zeros(1, dtype=torch.int32, device=dev)
        seq_len = torch.empty(1, dtype=torch.int32, device=dev)
        slot = torch.empty(1, dtype=torch.int32, device=dev)
        page_table = sh["table"].to(dev)
        cos_sin = ref.rope_table(length + 1).to(dev)

        def forward() -> None:
            layer.launch_gluon_muse_spark_layer_gfx950(
                weights,
                h=h_in,
                a=a_in,
                h_out=h_out,
                a_out=a_out,
                seq_len=seq_len,
                slot=slot,
                page_table=page_table,
                k_cache=k_cache,
                v_cache=v_cache,
                cos_sin=cos_sin,
                scratch=scratch,
                step=step,
                peer_bases=group.bases,
                layer=2,
                chain=1,
                whole_rows=False,
                tap=None,
                tap_slot=None,
                warm_next=None,
                proj_in=None,
                proj_out=None,
                trace_buf=None,
                stuck_rec=None,
                # Graph replays and eager launches repeat layer 2 between barriers.
                check_parity=False,
                compile_only=False,
            )
            # The seam moves the tag base once per forward, in the graph.
            step.add_(1)

        def refresh(n: int) -> None:
            pos = n - 1
            seq_len.fill_(n)
            slot.fill_(int(sh["table"][pos // 64]) * 64 + pos % 64)

        group.finish_prepare()
        refresh(length)
        forward()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        dist.barrier()
        for k, n in enumerate((length, length - 1, length - 70)):
            refresh(n)
            graph.replay()
            torch.cuda.synchronize()
            assert int(step) == 2 + 2 * k
            replayed = h_out.clone(), a_out.clone()
            dist.barrier()
            # An eager launch of the same step (under the next tag) gives
            # the same bits.
            forward()
            torch.cuda.synchronize()
            assert torch.equal(h_out, replayed[0])
            assert torch.equal(a_out, replayed[1])
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_graph_replays_match_eager_launches():
    """The decode graph captures the launch: a replay reads the step, the
    length and the write slot from device tensors, so each replay with
    refreshed values equals an eager launch of the same step."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_graph_worker, args=(port, 3000), nprocs=WORLD, join=True)


# (warm, own q | k | v, next layer's q | k | v) of the sliding and the full
# plan: the warm stage or not (the last fused layer has none), and past one
# row the launches of a run that hand the projection on (qnext).
_VARIANTS = [
    ((False, True, False), (False, True, False)),
    ((False, True, False), (True, True, False)),
    ((True, True, False), (False, True, False)),
    ((False, True, True), (False, False, True)),
    ((False, False, True), (False, False, False)),
    ((False, False, False), (False, True, True)),
]


@pytest.mark.parametrize("variants", _VARIANTS)
@pytest.mark.parametrize("check_parity", [False, True])
@pytest.mark.parametrize("rows", range(1, layer.MAX_ROWS + 1))
def test_both_kinds_of_layer_lay_the_scratch_out_alike(
    rows: int, variants, check_parity
):
    """Sliding and full launches share one scratch and alternate the peer
    regions' copies. A rank a launch ahead must then put only into the copy
    no slower rank still reads, which holds only if both plans place every
    region, and both copies of each peer region, at the same pairs, whatever
    their stages (the warm stage, qkvg or qnext)."""
    if rows == 1 and any(v[2] for v in variants):
        pytest.skip("qnext runs past one row")
    sliding, full = (
        layer.layer_plan(
            full=kind,
            rows=rows,
            chain=rows,
            warm_next=warm_next,
            own_qkvg=own_qkvg,
            next_qkvg=next_qkvg,
            check_parity=check_parity,
        )
        for kind, (warm_next, own_qkvg, next_qkvg) in zip((False, True), variants)
    )
    assert sliding.regions == full.regions
    assert sliding.peers == full.peers
    assert sliding.row_pairs == full.row_pairs
    for name, _ in sliding.regions:
        assert sliding.offset(name) == full.offset(name), name
    for name in sliding.peers:
        assert sliding.peer_copy(name) == full.peer_copy(name), name


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("rows", range(1, layer.MAX_ROWS + 1))
def test_peer_regions_split_into_rank_ordered_blocks_without_gaps(
    rows: int, full: bool
):
    """A launch with absent ranks (the layer bench's --gpus 1, 2 or 4) fills
    their share of every peer region from :func:`moe_tail.rank_blocks`: the
    blocks must cover each region's row exactly once, in order, and split
    evenly over the ranks, or the present ranks poll pairs nobody writes."""
    plan = layer.layer_plan(
        full=full,
        rows=rows,
        chain=rows,
        warm_next=False,
        own_qkvg=True,
        next_qkvg=False,
        check_parity=False,
    )
    sizes = dict(plan.regions)
    for name in plan.peers:
        at = 0
        for start, pairs in moe_tail.rank_blocks(name, sizes[name]):
            assert start == at, (name, start, at)
            assert pairs > 0 and pairs % WORLD == 0, (name, start, pairs)
            at = start + pairs
        assert at == sizes[name], (name, at, sizes[name])


# The layer bench with one present rank (--gpus 1): it writes the absent
# ranks' segments of every peer region before each launch. Polls give up
# after ~14 ms (11 << 17 ticks of the 100 MHz clock), far past any wait of a
# ~100 us launch, so a segment the bench misses fails the run at once
# instead of after a minute. The mp.spawn worker runs this file as its main
# module, so it shortens the timeout too, before it compiles.
_ONE_GPU_BENCH = textwrap.dedent("""
    import sys
    from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox

    mailbox.POLL_CHECK = 64
    mailbox.POLL_TIMEOUT_SHIFT = 17

    if __name__ == "__main__":
        import muse_spark_layer_bench

        sys.argv = ["muse_spark_layer_bench", *sys.argv[1:]]
        muse_spark_layer_bench.main()
    """)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("rows, requests", [(1, 1), (8, 1), (8, 2)])
def test_one_gpu_bench_launches_find_every_absent_rank_segment(
    tmp_path, full: bool, rows: int, requests: int
):
    """With fewer GPUs than ranks the bench stands in for the absent ranks:
    a launch must find every pair they would have written (each of ffn's
    rank-ordered blocks, every row), or a poll gives up and records what
    it waited for."""
    script = tmp_path / "one_gpu_bench.py"  # a spawned worker needs a file
    script.write_text(_ONE_GPU_BENCH)
    args = ["--gpus", "1", "--kv-dtype", "fp8", "--length", "3000"]
    args += ["--rows", str(rows), "--requests", str(requests)]
    args += ["--distinct", "0", "--replays", "1", "--stuck-dir", str(tmp_path)]
    args += ["--full"] if full else []
    # A cache of its own: the short timeout is in the compiled code. The
    # script runs from tmp_path, so this directory (the bench and its
    # helpers) goes on its path.
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.pathsep.join(p for p in (here, os.environ.get("PYTHONPATH")) if p)
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=path)
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=1200,
        env=env,
        cwd=os.path.dirname(__file__),
    )
    record = str(tmp_path / "rank0.rec")
    assert not stuck.records(stuck.read_file(record)), "\n".join(
        stuck_report.describe_file(record)
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_the_packer_names_every_tensor_of_another_layout():
    """The regular path may hold the fused projection of every rank (rank
    major) and stores the router transposed; the packer reports each such
    tensor instead of tiling the wrong rows."""
    f32, bf16 = torch.float32, torch.bfloat16
    row = torch.empty(HIDDEN, dtype=f32)
    pooled = torch.empty(POOLED, dtype=f32)
    experts = (torch.empty(256, 1),) * 4
    with pytest.raises(ValueError) as error:
        pack_layer(
            rank=0,
            full=False,
            qkvg=torch.empty(WORLD * QKVG, HIDDEN, dtype=bf16),
            o=torch.empty(HIDDEN, Q, dtype=bf16),
            alpha1=row,
            beta1=row,
            pre_gain=row,
            router=torch.empty(256, HIDDEN, dtype=f32),
            router_bias=torch.empty(256, dtype=f32),
            pool=torch.empty(POOLED, HIDDEN, dtype=bf16),
            experts=experts,
            pre_exp_gain=pooled,
            pen_gain=pooled,
            post=torch.empty(HIDDEN, POOLED, dtype=bf16),
            post_gain=row,
            alpha2=row,
            beta2=row,
            in_gain=row,
        )
    assert str(error.value) == (
        f"qkvg: expected torch.bfloat16 ({QKVG}, {HIDDEN}), got torch.bfloat16 "
        f"({WORLD * QKVG}, {HIDDEN}); router: expected torch.float32 ({HIDDEN}, "
        f"256), got torch.float32 (256, {HIDDEN})"
    )
