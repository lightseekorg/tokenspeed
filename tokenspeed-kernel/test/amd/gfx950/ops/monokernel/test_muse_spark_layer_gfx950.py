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
from datetime import timedelta

import muse_spark_reference as ref
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.benchmark import muse_spark_monokernel as inputs
from tokenspeed_kernel.benchmark import muse_spark_monokernel_gluon as gluon_inputs
from tokenspeed_kernel.numerics.torch.moe import torch_grouped_sigmoid_topk
from tokenspeed_kernel.numerics.torch.residual import torch_post_norm_residual_pre_norm
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
    cs = ref.rope_table(end + 1)
    b = torch.zeros(rows, HIDDEN)
    for r in range(WORLD):
        w = inputs.layer_rank_weights(r, end)
        qkvg = (sh["a_in"].float() @ w["w_qkvg"].float().T).to(torch.bfloat16)
        z, _, _, _, _, _ = ref.attention_z(
            qkvg,
            w["k_cache"],
            w["v_cache"],
            sh["table"],
            length,
            window,
            cs[length - 1 : end],
        )
        b = b + (z.float() @ w["w_o"].float().T).to(torch.bfloat16).float()
    # The attention boundary: h' = alpha1 h + beta1 norm(b), then the
    # pre-expert norm u (the shared reference, FP64).
    hp, u, _ = _boundary(sh, "1", sh["h"], b.to(torch.bfloat16), None, sh["pre_gain"])
    gated1 = (hp - sh["alpha1"].double() * sh["h"].double()).abs().float()
    hp, u = hp.float(), u.float()
    w_pool = torch.cat(
        [inputs.layer_rank_weights(r, end)["w_pool"] for r in range(WORLD)]
    )
    pooled = (u.to(torch.bfloat16).float() @ w_pool.float().T).to(torch.bfloat16)
    w_post = torch.cat(
        [inputs.layer_rank_weights(r, end)["w_post"] for r in range(WORLD)]
    )
    h2 = torch.empty(rows, HIDDEN)
    a = torch.empty(rows, HIDDEN, dtype=torch.bfloat16)
    gated = torch.empty(rows, HIDDEN)
    # The router: sigmoid scores, the bias picks, unbiased weights (one group).
    logits = (u.double() @ sh["router"].double().T) * 0.17677669529663687
    ids, route_weights = torch_grouped_sigmoid_topk(logits, sh["bias"], 8, 1, 1, 1.0)
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
        weights = route_weights[t]
        p = routes[t].to(torch.bfloat16).float() * sh["pen_gain"]
        p = p * torch.rsqrt(p.pow(2).mean(1, keepdim=True) + 1e-8)
        combined = (p * weights[:, None]).sum(0).to(torch.bfloat16)
        y = (combined.float() @ w_post.float().T).to(torch.bfloat16)
        # The FFN boundary from the kernel's h' (the shared reference).
        stream, _, a_t = _boundary(
            sh, "2", kernel_hp[t : t + 1], y[None], sh["post_gain"], sh["in_gain"]
        )
        h2[t], a[t] = stream[0].float(), a_t[0]
        gated[t] = (stream[0] - sh["alpha2"].double() * kernel_hp[t].double()).abs()
    return pooled, hp, gated1, h2, a, gated


def _boundary(sh, which, stream, branch, post_gain, pre_gain):
    """A layer boundary of the shared inputs ``sh`` with gate ``which`` ("1":
    attention, "2": FFN), by the shared reference: ``alpha stream + beta
    norm(branch * post_gain)``, then its norm times ``pre_gain``.

    Returns:
        ``(stream FP64, pre FP64, pre BF16)``.
    """
    return torch_post_norm_residual_pre_norm(
        stream=stream,
        branch=branch,
        post_weight=post_gain,
        post_eps=1e-8,
        post_weight_offset=0.0,
        alpha=sh["alpha" + which],
        beta=sh["beta" + which],
        pre_weight=pre_gain,
        pre_eps=1e-5,
        pre_weight_offset=0.0,
        enable_pdl=False,
        branch_sentinel=None,
        branch_ranks=1,
        branch_publish=None,
    )


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
    w = inputs.layer_rank_weights(rank, length)
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
        pool=torch.cat(
            [inputs.layer_rank_weights(r, length)["w_pool"] for r in range(WORLD)]
        ).to(dev),
        experts=experts,
        pre_exp_gain=sh["pre_exp_gain"].to(dev),
        pen_gain=sh["pen_gain"].to(dev),
        post=torch.cat(
            [inputs.layer_rank_weights(r, length)["w_post"] for r in range(WORLD)]
        ).to(dev),
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
        # A draft's taps: each rank's launch stores the whole residual rows
        # into slot 3 of its own copy.
        tap = ctx.zeros((rows, 5, HIDDEN), dtype=torch.bfloat16)
        end = length + rows - 1
        sh = inputs.layer_shared(42, length, rows)
        w = inputs.layer_rank_weights(rank, end)
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
        # The tap on every rank: each rank's owned h'' in its columns, rounded
        # once to BF16, the other slots untouched.
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
    tensors = _layer_tensors(0, dev, inputs.layer_shared(42, 3000, 1), 3000, False)
    tensors["experts_kernel"] = "triton_mxfp4_precomputed_moe_apply"
    with pytest.raises(ValueError, match="triton_mxfp4_precomputed_moe_apply"):
        prepare_muse_spark_layer(tensors=tensors, max_rows=layer.MAX_ROWS)


@pytest.mark.parametrize("solution", [None, "gluon"])
@pytest.mark.parametrize("full", [False, True])
def test_the_kernel_boundary_packs_a_layer_for_its_launches(full, solution):
    """The runtime packs a layer through the kernel boundary
    (``prepare_muse_spark_layer``). A forced solution selects the layer's
    prepare and decode kernels each of its own op: the gluon prepare packs
    the layer, and the gluon layer launch runs it."""
    from tokenspeed_kernel.ops.model.muse_spark.monokernel import (
        FusedDecodeLayer,
        prepare_muse_spark_layer,
    )

    dev = torch.device("cuda:0")
    tensors = _layer_tensors(0, dev, inputs.layer_shared(42, 3000, 1), 3000, full)
    tensors["experts_kernel"] = "gluon_mxfp4_precomputed_moe_apply"
    fused = prepare_muse_spark_layer(
        tensors=tensors, max_rows=layer.MAX_ROWS, solution=solution
    )

    assert isinstance(fused, FusedDecodeLayer)
    assert fused.weights.full is full and fused.weights.rank == 0
    assert fused.decode.name == "gluon_muse_spark_layer_gfx950"


def test_the_kernel_boundary_refuses_a_kernel_name_as_its_solution():
    """One forced solution selects both the prepare and the decode op: a
    kernel name (here the decode's) would hand the prepare op the decode
    kernel, and raises instead."""
    from tokenspeed_kernel.ops.model.muse_spark.monokernel import (
        prepare_muse_spark_layer,
    )
    from tokenspeed_kernel.selection import NoKernelFoundError

    dev = torch.device("cuda:0")
    tensors = _layer_tensors(0, dev, inputs.layer_shared(42, 3000, 1), 3000, False)
    tensors["experts_kernel"] = "gluon_mxfp4_precomputed_moe_apply"
    with pytest.raises(NoKernelFoundError):
        prepare_muse_spark_layer(
            tensors=tensors,
            max_rows=layer.MAX_ROWS,
            solution="gluon_muse_spark_layer_gfx950",
        )


@pytest.mark.parametrize("full", [False, True])
def test_one_binary_serves_every_layer_step_and_length(full):
    """What serving changes from launch to launch (the layer index, the step,
    the lengths, page-table widths and the KV slots) reaches the kernel at
    runtime: compiling the launches for one GPU's rank finds the first
    launch's binary. A new row count is a new plan, which the guard must
    see compile."""
    rows, length = 8, 900
    end = length + rows - 1
    dev = torch.device("cuda:0")
    sh = inputs.layer_shared(42, length, rows)
    w = inputs.layer_rank_weights(0, end)
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

    def compile_launch(
        index: int, step: int, seq_len: int, rows: int, width: int
    ) -> None:
        pos = torch.arange(seq_len - 1, seq_len - 1 + rows)
        slots = sh["table"][pos // 64].long() * 64 + pos % 64
        table = torch.zeros(1, width, dtype=torch.int32)
        table[0, : len(sh["table"])] = sh["table"]
        layer.launch_gluon_muse_spark_layer_gfx950(
            weights,
            h=torch.zeros(rows, OWNED, device=dev),
            a=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=dev),
            h_out=torch.empty(rows, OWNED, device=dev),
            a_out=torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev),
            seq_len=(pos + 1).to(torch.int32).to(dev),
            slot=slots.to(torch.int32).to(dev),
            page_table=table.to(dev),
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
    compile_launch(lay, 3, length, rows, 16)
    # Widths cross Triton's divisibility-by-16 specialization boundary.
    with assert_no_triton_compile(layer.gluon_muse_spark_layer_gfx950):
        for step, (index, seq_len, width) in enumerate(
            (
                (lay + 1, length, 16),
                (lay + 18, length - 1, 17),
                (lay + 35, length - 70, 31),
                (lay + 36, length, 48),
            ),
            start=4,
        ):
            compile_launch(index, step, seq_len, rows, width)
    # The guard is live: a row count the server did not warm compiles.
    with pytest.raises(AssertionError, match="compiled 1 new"):
        with assert_no_triton_compile(layer.gluon_muse_spark_layer_gfx950):
            compile_launch(lay, 8, length, rows - 1, 16)


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("full", [False, True])
def test_layer_matches_the_regular_path_math(full, rows):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, 3000, full, rows), nprocs=WORLD, join=True)


def _tap_context_worker(rank: int, port: int) -> None:
    """Five launches store a draft's five taps, then the draft's context write
    reads them; the same write on the taps the regular capture would hand it
    (each launch's whole h'' rows, rounded once to BF16) gives the same draft
    cache bytes."""
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import draft_context

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
        ctx = get_or_create_context(heap_size=512 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
        rows, length, taps_n = 8, 3000, 5
        plans = [
            layer.layer_plan(
                full=full,
                rows=rows,
                chain=rows,
                warm_next=True,
                own_qkvg=True,
                next_qkvg=False,
                check_parity=False,
            )
            for full in (False, True)
        ]
        scratch = ctx.zeros((max(p.scratch_pairs for p in plans),), dtype=torch.int64)
        tap = ctx.zeros((rows, taps_n, HIDDEN), dtype=torch.bfloat16)
        c_plan = draft_context.context_plan(rows)
        c_scratch = ctx.zeros((c_plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        end = length + rows - 1
        captured = []
        # The served draft's taps follow sliding and full layers alike.
        for slot, full in enumerate((False, True, False, True, False)):
            sh = inputs.layer_shared(40 + slot, length, rows)
            w = inputs.layer_rank_weights(rank, end)
            weights = _pack(rank, dev, sh, end, full)
            h_next = torch.empty(rows, OWNED, device=dev)
            pos = torch.arange(length - 1, length - 1 + rows)
            slots = sh["table"][pos // 64].long() * 64 + pos % 64
            for compile_only in (True, False):
                layer.launch_gluon_muse_spark_layer_gfx950(
                    weights,
                    h=sh["h"][:, own].contiguous().to(dev),
                    a=sh["a_in"].to(dev),
                    h_out=h_next,
                    a_out=torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev),
                    seq_len=(pos + 1).to(torch.int32).to(dev),
                    slot=slots.to(torch.int32).to(dev),
                    page_table=sh["table"].to(dev),
                    k_cache=w["k_cache"].to(dev),
                    v_cache=w["v_cache"].to(dev),
                    cos_sin=ref.rope_table(end + 1).to(dev),
                    scratch=scratch,
                    step=torch.full((1,), 3 + slot, dtype=torch.int32, device=dev),
                    peer_bases=group.bases,
                    layer=1 + slot,
                    chain=rows,
                    whole_rows=False,
                    tap=tap,
                    tap_slot=slot,
                    warm_next=weights,
                    proj_in=None,
                    proj_out=None,
                    trace_buf=None,
                    stuck_rec=None,
                    check_parity=False,
                    compile_only=compile_only,
                )
                torch.cuda.synchronize()
                dist.barrier()
            # What the regular capture hands the draft for this tap: the
            # whole h'' rows entering the next layer, rounded once to BF16.
            parts = [torch.empty_like(h_next.cpu()) for _ in range(WORLD)]
            dist.all_gather(parts, h_next.cpu())
            captured.append(torch.cat(parts, dim=1).to(torch.bfloat16))
        regular = torch.stack(captured, dim=1)  # [rows, taps, 8192]
        assert torch.equal(tap.cpu(), regular)

        # The draft's context write on the stored taps and on the captured
        # ones: the same kernel on the same bytes writes the same cache rows.
        g = torch.Generator(device=dev).manual_seed(7)

        def randn(*shape, std, gen=g):
            return (torch.randn(*shape, generator=gen, device=dev) * std).to(
                torch.bfloat16
            )

        fc = randn(HIDDEN, taps_n * HIDDEN, std=0.005)
        mine = torch.Generator(device=dev).manual_seed(700 + rank)
        weights = draft_context.pack_draft_context(
            {
                "fc": fc[rank * HIDDEN // WORLD : (rank + 1) * HIDDEN // WORLD],
                "hidden_norm": (1 + randn(HIDDEN, std=0.1).float()).to(torch.bfloat16),
                **{
                    f"qkv_{i}": randn(1280, HIDDEN, std=0.01, gen=mine)
                    for i in range(5)
                },
                **{
                    f"k_norm_{i}": (1 + randn(128, std=0.1, gen=mine).float()).to(
                        torch.bfloat16
                    )
                    for i in range(5)
                },
            },
            rank,
        )
        positions = torch.arange(length, length + rows, dtype=torch.int32, device=dev)
        cache_slots = torch.arange(100, 100 + rows, dtype=torch.int32, device=dev)
        # The draft's RoPE table is FP32 [positions, 128] (128-wide heads),
        # not the target layer's 64-wide one.
        inv = 1.0 / (500000.0 ** (torch.arange(0, 128, 2, device=dev).float() / 128))
        angle = torch.arange(length + rows + 1, device=dev).float()[:, None] * inv
        cos_sin = torch.cat([angle.cos(), angle.sin()], dim=-1)
        written = []
        for step, taps in (
            (3, tap.view(rows, -1)),
            (4, regular.to(dev).view(rows, -1)),
        ):
            k_caches = tuple(
                torch.zeros(512, 1, 128, dtype=torch.bfloat16, device=dev)
                for _ in range(5)
            )
            v_caches = tuple(torch.zeros_like(k) for k in k_caches)
            for compile_only in (True, False):
                draft_context.launch_gluon_muse_spark_draft_context_gfx950(
                    weights,
                    taps=taps.contiguous(),
                    positions=positions,
                    slots=cache_slots,
                    k_caches=k_caches,
                    v_caches=v_caches,
                    cos_sin=cos_sin,
                    scratch=c_scratch,
                    step=torch.full((1,), step, dtype=torch.int32, device=dev),
                    peer_bases=group.bases,
                    trace_buf=None,
                    compile_only=compile_only,
                )
                torch.cuda.synchronize()
                dist.barrier()
            written.append([c.cpu() for c in k_caches + v_caches])
        for stored, captured_ in zip(*written):
            assert torch.equal(stored, captured_)
        assert any(c.any() for c in written[0])
    finally:
        dist.destroy_process_group()


def test_the_launches_taps_feed_the_draft_context_as_the_capture_would():
    """The served draft's taps (rows of the residual entering layers 2, 17,
    31, 45 and 60) come from the launches' tap stores; the draft's context
    write must see exactly the bytes the regular capture would give it."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_tap_context_worker, args=(port,), nprocs=WORLD, join=True)


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
        # Request 0 is layer_shared's; request 1 has inputs and pages of its own.
        sh = inputs.layer_shared(42, lengths[0], rows)
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
        w = inputs.layer_rank_weights(rank, 3 * max(ends))
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
        sh = inputs.layer_shared(42, length, rows)
        w = inputs.layer_rank_weights(rank, end)
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


def _serving_order_worker(rank: int, port: int) -> None:
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
        sh = inputs.layer_shared(42, length, rows)
        w = inputs.layer_rank_weights(rank, end)
        packed = _pack(rank, dev, sh, end, False)
        # A step's last two launching layers, then the next step's first two:
        # (layer index, step offset, full). Serving launches layers 1-60
        # every step, so the parity (layer & 1) alternates across the step
        # boundary too: 59, 60, then 1, 2.
        order = [(59, 0, False), (60, 0, True), (1, 1, False), (2, 1, False)]
        layers = []
        for i, (_, _, full) in enumerate(order):
            g = torch.Generator().manual_seed(3000 + 10 * rank + i)
            qkvg = (torch.randn(QKVG, HIDDEN, generator=g) / HIDDEN**0.5).to(
                torch.bfloat16
            )
            layers.append(
                dataclasses.replace(packed, full=full, qkvg=bf16_tiles(qkvg.to(dev)))
            )
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        pos = torch.arange(length - 1, length - 1 + rows)
        slots = (sh["table"][pos // 64].long() * 64 + pos % 64).to(torch.int32)
        seq_len = (pos + 1).to(torch.int32).to(dev)
        slot = slots.to(dev)
        page_table = sh["table"].to(dev)
        cos_sin = ref.rope_table(end + 1).to(dev)
        proj = torch.zeros(2, rows, int(PROJ_WORDS), dtype=torch.int32, device=dev)
        fused = {index for index, _, _ in order}

        def run(first_step: int, caches, barriered: bool):
            # The step counter is a device tensor that moves once per step on
            # the stream, after the step's last launch, as in serving.
            step = torch.full((1,), first_step, dtype=torch.int32, device=dev)
            h, a = sh["h"][:, own].contiguous().to(dev), sh["a_in"].to(dev)
            out = []
            for i, (index, offset, _) in enumerate(order):
                if i and offset != order[i - 1][1]:
                    step.add_(1)
                # The serving hand-off of the next layer's projection.
                proj_in = proj[index & 1] if index - 1 in fused else None
                proj_out = proj[(index + 1) & 1] if index + 1 in fused else None
                h_out = torch.empty(rows, OWNED, device=dev)
                a_out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
                layer.launch_gluon_muse_spark_layer_gfx950(
                    layers[i],
                    h=h,
                    a=a,
                    h_out=h_out,
                    a_out=a_out,
                    seq_len=seq_len,
                    slot=slot,
                    page_table=page_table,
                    k_cache=caches[i][0],
                    v_cache=caches[i][1],
                    cos_sin=cos_sin,
                    scratch=scratch,
                    step=step,
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
                    # Traps if two launches in a row share a parity.
                    check_parity=True,
                    compile_only=False,
                )
                if barriered:
                    torch.cuda.synchronize()
                    dist.barrier()
                out.append((h_out, a_out))
                h, a = h_out, a_out
            torch.cuda.synchronize()
            dist.barrier()
            return [(h.cpu(), a.cpu()) for h, a in out]

        group.finish_prepare()
        caches = [[w["k_cache"].to(dev), w["v_cache"].to(dev)] for _ in order]
        barriered = [[c.clone() for c in cs] for cs in caches]
        unbarriered = [[c.clone() for c in cs] for cs in caches]
        want = run(3, barriered, True)
        # The same launches back to back with nothing between them, under the
        # next steps' tags: a rank may start the next step's first launch
        # while its peers still finish the last one.
        got = run(5, unbarriered, False)
        for i, (index, _, _) in enumerate(order):
            assert torch.equal(got[i][0], want[i][0]), f"layer {index}: residual"
            assert torch.equal(got[i][1], want[i][1]), f"layer {index}: next input"
            for c in range(2):
                assert torch.equal(
                    unbarriered[i][c], barriered[i][c]
                ), f"layer {index}: cache {c}"
    finally:
        dist.destroy_process_group()


def test_launches_across_a_step_boundary_run_unbarriered():
    """Serving runs the launches back to back, with no synchronization
    between them: layer 60 of one step and layer 1 of the next run under
    alternating parity and give the same bits as the same launches with a
    barrier after each (the parity check traps on a repeated parity)."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_serving_order_worker, args=(port,), nprocs=WORLD, join=True)


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
        sh = inputs.layer_shared(42, length, 1)
        weights = _pack(rank, dev, sh, length, False)
        w = inputs.layer_rank_weights(rank, length)
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


@pytest.mark.parametrize("gpus", [1, 2, 4])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("rows, requests", [(1, 1), (8, 1), (16, 2)])
def test_the_bench_stands_in_for_the_absent_ranks_flags_as_raised(
    gpus: int, full: bool, rows: int, requests: int
):
    """The layer bench writes ``payload | tag << 32`` into the absent ranks'
    segments (``tokenspeed_kernel.benchmark.muse_spark_monokernel_gluon``)
    of every peer region before a launch. A flag pair there must read
    exactly ``(tag << 32) | 1``, as ``raise_flag_peers`` writes it, from
    the first launch on: ``Mailbox.poll_flags`` waits for that value, so a
    ghost flag of payload 0 stops the launch."""
    plan = layer.layer_plan(
        full=full,
        rows=rows,
        chain=rows // requests,
        warm_next=False,
        own_qkvg=True,
        next_qkvg=False,
        check_parity=False,
    )
    assert "xflag" in plan.flags and ("ffnf" in plan.flags) == (rows > 8)
    tag = 0x1234ABCD
    sizes = dict(plan.regions)
    for parity in (0, 1):
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64)
        # The fill kernel's store, before the first launch and after settling.
        for payloads in (
            None,
            torch.full_like(gluon_inputs.absent_index(plan, parity, gpus), 7),
        ):
            ghost = gluon_inputs.ghost_payloads(plan, parity, gpus, payloads)
            scratch[gluon_inputs.absent_index(plan, parity, gpus)] = ghost | (tag << 32)
            for name in plan.flags:
                for row in range(rows):
                    base = (
                        plan.offset(name)
                        + parity * plan.peer_copy(name)
                        + row * plan.row_pairs
                    )
                    for start, pairs in moe_tail.rank_blocks(name, sizes[name]):
                        seg = pairs // WORLD
                        for rank in range(gpus, WORLD):
                            at = base + start + rank * seg
                            assert torch.all(
                                scratch[at : at + seg] == (tag << 32) | 1
                            ), (name, row, rank)


def _absent_ranks_plan(full: bool, rows: int, requests: int):
    """The plan of :func:`_absent_ranks_worker`'s launches."""
    return layer.layer_plan(
        full=full,
        rows=requests * rows,
        chain=rows,
        warm_next=False,
        own_qkvg=True,
        next_qkvg=False,
        check_parity=False,
    )


def _absent_ranks_worker(
    rank: int, port: int, full: bool, rows: int, requests: int, record: str
) -> None:
    """One process (rank 0) launches the TP8 layer through the public
    launcher while the host stands in for ranks 1 to 7: their heap bases
    point at a local ghost buffer, and before each launch their segment of
    every peer region holds the launch's tag (flags raised), as
    ``muse_spark_monokernel_gluon`` writes it for the layer bench."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=1,
        timeout=timedelta(seconds=900),
    )
    try:
        dev = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
        n = requests * rows
        plan = _absent_ranks_plan(full, rows, requests)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        ghost = torch.zeros_like(scratch)
        bases = torch.empty(WORLD, dtype=torch.int64, device=dev)
        bases[:1] = group.bases.to(dev)
        bases[1:] = int(group.bases[rank]) + (ghost.data_ptr() - scratch.data_ptr())
        index = [gluon_inputs.absent_index(plan, p, 1).to(dev) for p in (0, 1)]
        bits = [gluon_inputs.ghost_payloads(plan, p, 1).to(dev) for p in (0, 1)]
        buf = stuck.stuck_buffer(256, 8, record, scratch.data_ptr(), rank)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        # Request b at length 3000 + 100 b, each with pages of its own.
        lengths = [3000 + 100 * b for b in range(requests)]
        ends = [length + rows - 1 for length in lengths]
        sh = inputs.layer_shared(42, lengths[0], n)
        tables, first = [], 1
        for end in ends:
            pages = (end + 63) // 64
            tables.append(torch.arange(first, first + pages, dtype=torch.int32))
            first += pages
        table = torch.zeros(requests, max(len(t) for t in tables), dtype=torch.int32)
        for b, t in enumerate(tables):
            table[b, : len(t)] = t
        pos = torch.cat([torch.arange(m - 1, m - 1 + rows) for m in lengths])
        slot = torch.cat(
            [
                t[q // 64].long() * 64 + q % 64
                for t, q in zip(tables, pos.view(requests, rows))
            ]
        )
        w = inputs.layer_rank_weights(rank, 64 * first)
        weights = _pack(rank, dev, sh, max(ends), full)
        step = torch.full((1,), 5, dtype=torch.int32, device=dev)
        group.finish_prepare()
        # Two launches, so both copies of the peer regions are read.
        for lay in (1, 2):
            step.add_(1)
            gluon_inputs.fill_absent_ranks(
                scratch, index[lay & 1], bits[lay & 1], step, lay
            )
            layer.launch_gluon_muse_spark_layer_gfx950(
                weights,
                h=sh["h"][:, own].contiguous().to(dev),
                a=sh["a_in"].to(dev),
                h_out=torch.empty(n, OWNED, device=dev),
                a_out=torch.empty(n, HIDDEN, dtype=torch.bfloat16, device=dev),
                seq_len=(pos + 1).to(torch.int32).to(dev),
                slot=slot.to(torch.int32).to(dev),
                page_table=table.to(dev),
                k_cache=w["k_cache"].to(dev),
                v_cache=w["v_cache"].to(dev),
                cos_sin=inputs.rope_table(max(ends) + 1).to(dev),
                scratch=scratch,
                step=step,
                peer_bases=bases,
                layer=lay,
                chain=rows,
                whole_rows=False,
                tap=None,
                tap_slot=None,
                warm_next=None,
                proj_in=None,
                proj_out=None,
                trace_buf=None,
                stuck_rec=buf.tensor,
                check_parity=False,
                compile_only=False,
            )
            torch.cuda.synchronize()
        stuck.close(buf, remove=False)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("rows, requests", [(1, 1), (8, 1), (8, 2)])
def test_one_gpu_launches_find_every_absent_rank_segment(
    tmp_path, full: bool, rows: int, requests: int
):
    """With one GPU standing in for the TP8 group, a launch must find every
    pair the absent ranks would have written (each peer region's
    rank-ordered blocks, the flags past 8 rows, every row), or a poll gives
    up, records what it waited for and stops the kernel."""
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    record = str(tmp_path / "rank0.rec")
    # A poll that gives up stops the worker; its record outlives it.
    try:
        mp.spawn(
            _absent_ranks_worker,
            args=(port, full, rows, requests, record),
            nprocs=1,
            join=True,
        )
    finally:
        if os.path.exists(record):
            t = stuck.read_file(record)
            plan = _absent_ranks_plan(full, rows, requests)
            assert not stuck.records(t), "\n".join(
                stuck.summarize(t, lambda index: [plan])
            )


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


def test_the_packer_refuses_experts_past_a_buffer_loads_offset():
    """The expert stages address the expert tensors with 32-bit buffer-load
    offsets: a tensor whose bytes reach past 2^31 is refused, the last byte
    of one that ends exactly there is accepted."""
    f32, bf16, u8 = torch.float32, torch.bfloat16, torch.uint8
    row = torch.empty(HIDDEN, dtype=f32)
    pooled = torch.empty(POOLED, dtype=f32)
    small = torch.empty(256, 1, dtype=u8, device="meta")
    # 256 experts of 2^23 bytes end exactly at 2^31; one more byte is past it.
    at_bound = torch.empty(256, 1 << 23, dtype=u8, device="meta")
    past = torch.empty(256, (1 << 23) + 1, dtype=u8, device="meta")

    def refusal(w13):
        try:
            pack_layer(
                rank=0,
                full=False,
                qkvg=torch.empty(QKVG, HIDDEN, dtype=bf16, device="meta"),
                o=torch.empty(HIDDEN, Q, dtype=bf16, device="meta"),
                alpha1=row,
                beta1=row,
                pre_gain=row,
                # Transposed on purpose: the packer always refuses, before it
                # tiles anything, and lists every tensor it refuses.
                router=torch.empty(256, HIDDEN, dtype=f32),
                router_bias=torch.empty(256, dtype=f32),
                pool=torch.empty(POOLED, HIDDEN, dtype=bf16, device="meta"),
                experts=(w13, small, small, small),
                pre_exp_gain=pooled,
                pen_gain=pooled,
                post=torch.empty(HIDDEN, POOLED, dtype=bf16, device="meta"),
                post_gain=row,
                alpha2=row,
                beta2=row,
                in_gain=row,
            )
        except ValueError as error:
            return str(error)
        return ""

    assert "w13 spans past 2^31 bytes" in refusal(past)
    assert "w13" not in refusal(at_bound)


@pytest.mark.parametrize("rows", [layer.STUCK_ROWS - 1, 256 + 1])
def test_the_launcher_refuses_a_stuck_buffer_without_a_record_per_wave(rows):
    """A stuck-poll buffer must hold a record for every wave of the launch
    (256 CTAs of 8 waves) and the header row: a per-CTA buffer is refused
    before anything compiles."""
    f32, bf16, i32 = torch.float32, torch.bfloat16, torch.int32
    with pytest.raises(ValueError, match="stuck_rec"):
        layer.launch_gluon_muse_spark_layer_gfx950(
            None,
            h=torch.zeros(1, 1024, dtype=f32),
            a=torch.zeros(1, 8192, dtype=bf16),
            h_out=torch.zeros(1, 1024, dtype=f32),
            a_out=torch.zeros(1, 8192, dtype=bf16),
            seq_len=torch.ones(1, dtype=i32),
            slot=torch.zeros(1, dtype=i32),
            page_table=torch.zeros(1, 16, dtype=i32),
            k_cache=torch.zeros(16, 64, 2, 64, dtype=bf16),
            v_cache=torch.zeros(16, 64, 2, 64, dtype=bf16),
            cos_sin=torch.zeros(64, 64, dtype=f32),
            scratch=None,
            step=None,
            peer_bases=None,
            layer=1,
            chain=1,
            whole_rows=False,
            tap=None,
            tap_slot=None,
            warm_next=None,
            proj_in=None,
            proj_out=None,
            trace_buf=None,
            stuck_rec=torch.zeros(rows, 8, dtype=torch.int64),
            check_parity=False,
            compile_only=True,
        )
