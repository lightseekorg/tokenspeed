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

"""The DFlash2 draft's context write (Muse Spark) in one launch across the 8
ranks: fc and hidden_norm of the target's taps, then every draft layer's K
and V of the verify rows into the draft cache."""

from __future__ import annotations

import contextlib
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

import muse_spark_standins as standins  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    draft_context,
)

WORLD, HIDDEN, TAPS_K, LAYERS, HEAD, ROWS = 8, 8192, 40960, 5, 128, 8
EPS = 1e-5
OWNED = HIDDEN // WORLD
ALL = tuple(range(WORLD))


def _rope_table(n: int, device) -> torch.Tensor:
    inv = 1.0 / (500000.0 ** (torch.arange(0, HEAD, 2, device=device).float() / HEAD))
    f = torch.arange(n, device=device).float()[:, None] * inv[None, :]
    return torch.cat([f.cos(), f.sin()], dim=-1)


def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _reference(taps, fc, norm_w, qkv, k_norm, cos_sin, positions):
    """``project_target_hidden`` then the fused context K/V write's math:
    each layer's K (RMSNorm, NeoX RoPE in FP32, one rounding) and V."""
    ctx = (taps.float() @ fc.float().T).to(torch.bfloat16)
    ctx = _rms(ctx, norm_w).to(torch.bfloat16)
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, : HEAD // 2], cs[:, HEAD // 2 :]
    out = []
    for layer in range(LAYERS):
        kv = (ctx.float() @ qkv[layer][1024:].float().T).to(torch.bfloat16)
        k, v = kv[:, :HEAD], kv[:, HEAD:]
        n = _rms(k, k_norm[layer])
        n1, n2 = n[:, : HEAD // 2], n[:, HEAD // 2 :]
        k = torch.cat([n1 * cos - n2 * sin, n2 * cos + n1 * sin], -1)
        out.append((k.to(torch.bfloat16), v))
    return out


def _worker(me: int, port: int, seed: int, present) -> None:
    """``present``: None for the whole TP8 group, one process a rank; or the
    two rank programs that run (process ``me`` runs ``present[me]``). The
    launches checked against the reference run a zero-shard model (the
    other ranks' fc rows are zero, so they put exactly zero, which the host
    writes before each launch); the two-requests launches, compared with
    each other, read keyed stand-ins for the other ranks
    (:func:`muse_spark_standins.standin_fill`)."""
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
        # Two requests' plan is the larger: one scratch serves both.
        plan = draft_context.context_plan(2 * ROWS)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        if present is None:
            rank, bases = me, group.bases
        else:
            rank = present[me]
            ghost = torch.zeros_like(scratch)
            bases = standins.peer_bases(group.bases, present, me, scratch, ghost)

        def stand_ins(step: int, rows: list[tuple[int, int]], zero: bool) -> None:
            if present is not None:
                launch_plan = draft_context.context_plan(len(rows))
                standins.standin_fill(
                    scratch, launch_plan, present, rows, step, 0, step & 1, zero
                )

        group.finish_prepare()
        g = torch.Generator(device=device).manual_seed(seed)

        def randn(*shape, std, gen=g):
            return (torch.randn(*shape, generator=gen, device=device) * std).to(
                torch.bfloat16
            )

        taps = randn(ROWS, TAPS_K, std=1.0)
        fc = randn(HIDDEN, TAPS_K, std=0.005)
        if present is not None:
            # A zero-shard model: the absent ranks' fc rows (the columns
            # they own) are zero, so their sums and normed columns are zero.
            fc[standins.absent_columns(HIDDEN, OWNED, present).to(device)] = 0
        norm_w = (1 + randn(HIDDEN, std=0.1).float()).to(torch.bfloat16)
        mine = torch.Generator(device=device).manual_seed(seed * 100 + rank + 1)
        qkv = [randn(1280, HIDDEN, std=0.01, gen=mine) for _ in range(LAYERS)]
        k_norm = [
            (1 + randn(HEAD, std=0.1, gen=mine).float()).to(torch.bfloat16)
            for _ in range(LAYERS)
        ]
        weights = draft_context.pack_draft_context(
            {
                "fc": fc[rank * HIDDEN // 8 : (rank + 1) * HIDDEN // 8],
                "hidden_norm": norm_w,
                **{f"qkv_{i}": qkv[i] for i in range(LAYERS)},
                **{f"k_norm_{i}": k_norm[i] for i in range(LAYERS)},
            },
            rank,
        )
        cos_sin = _rope_table(70000, device)
        slots_n = 4096
        # The rows' slots lie past 2^31 cache elements (a slot is HEAD
        # elements): a slot offset computed in int32 writes the wrong place.
        # Only those last slots are written and read.
        far = (1 << 31) // HEAD + 1

        def far_cache() -> torch.Tensor:
            c = torch.empty(far + slots_n, 1, HEAD, dtype=torch.bfloat16, device=device)
            c[far:].zero_()
            return c

        k_caches = tuple(far_cache() for _ in range(LAYERS))
        v_caches = tuple(far_cache() for _ in range(LAYERS))
        for step, start in ((3, 60000), (4, 129)):
            positions = torch.arange(
                start, start + ROWS, dtype=torch.int32, device=device
            )
            slots = far + torch.randperm(slots_n, generator=g, device=device)[:ROWS]
            slots = slots.to(torch.int32)
            stand_ins(step, [(0, t) for t in range(ROWS)], True)
            # The second launch, at other rows, runs the first one's binary.
            with (
                assert_no_triton_compile(
                    draft_context.gluon_muse_spark_draft_context_gfx950
                )
                if step == 4
                else contextlib.nullcontext()
            ):
                draft_context.launch_gluon_muse_spark_draft_context_gfx950(
                    weights,
                    taps=taps,
                    positions=positions,
                    slots=slots,
                    k_caches=k_caches,
                    v_caches=v_caches,
                    cos_sin=cos_sin,
                    scratch=scratch,
                    step=torch.full((1,), step, dtype=torch.int32, device=device),
                    peer_bases=bases,
                    trace_buf=None,
                    compile_only=False,
                )
            torch.cuda.synchronize()
            want = _reference(taps, fc, norm_w, qkv, k_norm, cos_sin, positions)
            for layer, (k, v) in enumerate(want):
                for got, ref in ((k_caches[layer], k), (v_caches[layer], v)):
                    got = got[slots, 0].float()
                    torch.testing.assert_close(
                        got,
                        ref.float(),
                        rtol=2**-6,
                        atol=ref.float().abs().max().item() * 2**-8,
                    )
            dist.barrier()
        # Two requests' verify rows in one launch write the bits that a launch
        # of each request's rows writes (the rows are independent).
        taps_b = randn(ROWS, TAPS_K, std=1.0)
        positions = torch.cat(
            [
                torch.arange(5000, 5000 + ROWS, dtype=torch.int32, device=device),
                torch.arange(64000, 64000 + ROWS, dtype=torch.int32, device=device),
            ]
        )
        slots = torch.randperm(slots_n, generator=g, device=device, dtype=torch.int32)[
            : 2 * ROWS
        ]
        caches = {}
        # The stand-ins are keyed by request and row: each launch's absent
        # ranks put the same values for the same request's row.
        keys = [(0, t) for t in range(ROWS)] + [(1, t) for t in range(ROWS)]
        for name, rows, step in (
            ("a", slice(0, ROWS), 5),
            ("b", slice(ROWS, 2 * ROWS), 6),
            ("both", slice(0, 2 * ROWS), 7),
        ):
            stand_ins(step, keys[rows], False)
            ks = tuple(
                torch.zeros(slots_n, 1, HEAD, dtype=torch.bfloat16, device=device)
                for _ in range(LAYERS)
            )
            vs = tuple(torch.zeros_like(c) for c in ks)
            draft_context.launch_gluon_muse_spark_draft_context_gfx950(
                weights,
                taps=torch.cat([taps, taps_b])[rows].contiguous(),
                positions=positions[rows],
                slots=slots[rows],
                k_caches=ks,
                v_caches=vs,
                cos_sin=cos_sin,
                scratch=scratch,
                step=torch.full((1,), step, dtype=torch.int32, device=device),
                peer_bases=bases,
                trace_buf=None,
                compile_only=False,
            )
            caches[name] = (ks, vs)
        torch.cuda.synchronize()
        for layer in range(LAYERS):
            for kind in (0, 1):
                both = caches["both"][kind][layer]
                for name, rows in (("a", slice(0, ROWS)), ("b", slice(ROWS, None))):
                    at = slots[rows]
                    assert torch.equal(both[at], caches[name][kind][layer][at])
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seed", [0])
def test_draft_context_write_in_one_launch_matches_fc_norm_and_kv(seed: int):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, seed, None), nprocs=WORLD, join=True)


@pytest.mark.parametrize("present", standins.PAIRS, ids=standins.PAIR_IDS)
@pytest.mark.parametrize("seed", [0])
def test_draft_context_write_in_one_launch_matches_fc_norm_and_kv_on_two_gpus(
    seed: int, present
):
    """As :func:`test_draft_context_write_in_one_launch_matches_fc_norm_and_kv`
    with the production rank programs ``present`` on two GPUs: K and V
    against the reference in a zero-shard model (the other ranks' fc rows
    are zero; the host writes their zero sums and columns), and two
    requests in one launch against a launch each with keyed stand-ins for
    the other ranks. Two non-zero terms per sum cannot check the rank order
    of the eight-rank sum of squares (see muse_spark_standins)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, seed, present), nprocs=2, join=True)


def test_one_binary_serves_every_step_position_and_slot():
    """Each launch changes the step, the rows' positions and their KV slots,
    all device tensors: compiling the launches for one GPU's rank finds the
    first launch's binary. Another request count is another plan, which the
    guard must see compile."""
    device = torch.device("cuda:0")
    g = torch.Generator(device=device).manual_seed(0)

    def randn(*shape, std):
        return (torch.randn(*shape, generator=g, device=device) * std).to(
            torch.bfloat16
        )

    weights = draft_context.pack_draft_context(
        {
            "fc": randn(HIDDEN // WORLD, TAPS_K, std=0.005),
            "hidden_norm": randn(HIDDEN, std=0.1),
            **{f"qkv_{i}": randn(1280, HIDDEN, std=0.01) for i in range(LAYERS)},
            **{f"k_norm_{i}": randn(HEAD, std=0.1) for i in range(LAYERS)},
        },
        0,
    )
    plan = draft_context.context_plan(2 * ROWS)
    # Compiling alone reads none of these: plain device memory stands in for
    # the symmetric heap and the peers.
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device=device)
    peer_bases = torch.zeros(WORLD, dtype=torch.int64, device=device)
    caches = tuple(
        torch.zeros(4096, 1, HEAD, dtype=torch.bfloat16, device=device)
        for _ in range(2 * LAYERS)
    )
    cos_sin = _rope_table(70000, device)

    def compile_launch(step: int, start: int, requests: int) -> None:
        rows = requests * ROWS
        draft_context.launch_gluon_muse_spark_draft_context_gfx950(
            weights,
            taps=randn(rows, TAPS_K, std=1.0),
            positions=torch.arange(
                start, start + rows, dtype=torch.int32, device=device
            ),
            slots=torch.randperm(4096, generator=g, device=device, dtype=torch.int32)[
                :rows
            ],
            k_caches=caches[:LAYERS],
            v_caches=caches[LAYERS:],
            cos_sin=cos_sin,
            scratch=scratch,
            step=torch.full((1,), step, dtype=torch.int32, device=device),
            peer_bases=peer_bases,
            trace_buf=None,
            compile_only=True,
        )

    compile_launch(3, 60000, 1)
    with assert_no_triton_compile(draft_context.gluon_muse_spark_draft_context_gfx950):
        for step, start in ((4, 129), (5, 0), (6, 65530)):
            compile_launch(step, start, 1)
    # The guard is live: two requests' rows are a new plan and compile.
    with pytest.raises(AssertionError, match="compiled 1 new"):
        with assert_no_triton_compile(
            draft_context.gluon_muse_spark_draft_context_gfx950
        ):
            compile_launch(7, 5000, 2)
