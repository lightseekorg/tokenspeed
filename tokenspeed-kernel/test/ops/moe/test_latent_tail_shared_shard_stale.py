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

"""The latent tail's ``shared_shard`` carries ``max_m`` rows; only ``m`` live.

``CollectiveKernel`` allocates its reduce-scatter destination with
``torch.empty((max_m, hidden_dim))`` and never zeroes it
(``allreduce_rmsnorm_reduce_scatter_early_exit.py:887-892``); the RS loop
writes rows ``[0, m)`` only (``:239-240``). The up-projection then validates
that buffer against ``(max_m, shard_dim)`` -- ``max_m``, not ``m``
(``fused_add_multicast_gemm.py:1224-1228``) -- while the latent it multiplies
is sliced to ``m`` rows (``:991``). Rows ``[m, max_m)`` therefore hold the
previous decode step's shared-expert output, or uninitialised memory on the
first call.

Correctness rests entirely on the GEMM epilogue's row bound, on two different
backends: a static-M SIMT skinny GEMM at ``m <= 5`` and a dynamic-M
tensor-core GEMM at ``m >= 6``. These tests poison rows ``[m, max_m)`` and
require bit-identical live rows, on both backends, eagerly and under
CUDA-graph replay. They also check the *write* side: the Lamport mailbox must
be left entirely at its empty sentinel, because a stray epilogue store past
row ``m`` would be read as already-arrived data by the next, larger-``m``
step -- stale shared output inside a live token, with no error anywhere.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

H, L, EPS = 7168, 3584, 1e-6
MAX_M = 16
# ``latent_tail._SKINNY_MAX_NUM_TOKENS`` is 5: m <= 5 compiles a static-M SIMT
# skinny GEMM (one CUBIN per m), m >= 6 uses the dynamic-M MMA GEMM whose row
# bound is a runtime predicate instead of a compile-time loop trip count.
# Straddle the switch and cover the whole capture ladder inside [1, 16].
M_VALUES = [1, 2, 3, 4, 5, 6, 8, 12, 16]
GRAPH_SIZES = [1, 2, 4, 5, 6, 8, 16]
LAMPORT_EMPTY_I32 = -0x80000000  # primitives.NEG_ZERO_F32_BITS, signed


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {8, 16},
    reason="launch with torchrun world size 8 or 16",
)


def _setup():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, torch.device("cuda", rank)


def _op(device):
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    if not latent_tail_supported(
        tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
    ):
        pytest.skip("platform does not support the multicast tail")
    return KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=device,
    )


def _uniform(shape, seed, device, scale=0.1):
    """Rank-independent random tensor: every rank builds the same bytes.

    The mailbox gathers one shard per rank, so making all ranks agree turns
    the distributed result into a locally computable reference.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(*shape, generator=g, dtype=torch.float32, device=device) * scale
    return x.to(torch.bfloat16).contiguous()


def _bits(t: torch.Tensor) -> torch.Tensor:
    """NaN-safe bitwise view: ``torch.equal`` on floats is false for NaN."""
    return t.contiguous().view(torch.int16)


def _sync(device):
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device.index])
    torch.cuda.synchronize()


def _run_up_projection(op, device, rank, m, live, tail_fill, latent, up_w):
    """``up_projection -> lamport_copy`` with a hand-built ``max_m``-row shard.

    Bypasses the collective so the shard's tail rows are exactly what this
    test wrote, rather than whatever the previous reduce-scatter left there.
    """
    shard = H // _world_size()
    shared_full = torch.empty(MAX_M, H, dtype=torch.bfloat16, device=device)
    shared_full.fill_(tail_fill)
    shared_full[:m].copy_(live)
    view = shared_full[:, rank * shard : (rank + 1) * shard]
    assert view.shape == (MAX_M, shard) and view.stride() == (H, 1)

    op._up_projection.ensure_compiled(m)
    mailbox = op._up_projection(
        latent[:m].contiguous(),
        up_w.narrow(0, rank * shard, shard),
        view,
    )
    out = op._lamport_copy(mailbox, m=m).squeeze(0)
    torch.cuda.synchronize()
    return out, mailbox


@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("fill", [float("nan"), float("-inf"), 3.0e4])
def test_poisoned_shard_tail_does_not_reach_live_rows(m, fill):
    """Rows ``[m, 16)`` filled with poison must not perturb rows ``[0, m)``.

    Bitwise equality, not a tolerance: the two runs differ only in bytes the
    kernel has no business reading, so any difference at all means the
    epilogue's predication uses the padded tile extent instead of the live
    ``m``.
    """
    rank, device = _setup()
    op = _op(device)
    latent = _uniform((MAX_M, L), 4242, device)
    up_w = _uniform((H, L), 4243, device, scale=0.02)
    live = _uniform((m, H), 5000 + m, device)

    base, _ = _run_up_projection(op, device, rank, m, live, 0.0, latent, up_w)
    base = base.clone()
    _sync(device)
    poisoned, _ = _run_up_projection(op, device, rank, m, live, fill, latent, up_w)
    _sync(device)

    assert torch.equal(_bits(base), _bits(poisoned)), (
        f"m={m} ({'skinny' if m <= 5 else 'dynamic MMA'} backend): the fused "
        f"up-projection's output depends on shared_shard rows [{m}, {MAX_M}), "
        "which hold the previous decode step's shared-expert output. "
        f"{int((_bits(base) != _bits(poisoned)).sum())} of {base.numel()} "
        "output elements changed."
    )


@pytest.mark.parametrize("m", M_VALUES)
def test_live_shard_rows_are_actually_consumed(m):
    """Positive control for the poison tests above.

    If the epilogue silently dropped the shared addend, every poison test
    would pass vacuously. Perturbing a *live* row must change the output.
    """
    rank, device = _setup()
    op = _op(device)
    latent = _uniform((MAX_M, L), 4242, device)
    up_w = _uniform((H, L), 4243, device, scale=0.02)
    live = _uniform((m, H), 5000 + m, device)

    base, _ = _run_up_projection(op, device, rank, m, live, 0.0, latent, up_w)
    base = base.clone()
    _sync(device)
    live2 = live.clone()
    live2[m - 1] += 1.0
    moved, _ = _run_up_projection(op, device, rank, m, live2, 0.0, latent, up_w)
    _sync(device)

    assert not torch.equal(_bits(base), _bits(moved)), (
        f"m={m}: perturbing live shard row {m - 1} did not change the output, "
        "so the shared addend is not reaching the epilogue and the staleness "
        "tests in this module are vacuous"
    )
    # ...and the result must still be the real product, not zeros.
    ref = (latent[:m].float() @ up_w.float().T + live.float()).to(torch.bfloat16)
    scale = max(float(ref.float().abs().max()), 1.0)
    err = float((base.float() - ref.float()).abs().max())
    assert err < 0.05 * scale, f"m={m}: up-projection err {err} vs scale {scale}"

    # Poison EVERY row, live ones included: the output must go non-finite.
    # This is the direct proof that the kernel reads this buffer at all, so a
    # clean tail-only result means "rows >= m are not read" rather than "the
    # shared addend is ignored".
    all_nan = torch.full((m, H), float("nan"), dtype=torch.bfloat16, device=device)
    every, _ = _run_up_projection(
        op, device, rank, m, all_nan, float("nan"), latent, up_w
    )
    _sync(device)
    assert bool((~torch.isfinite(every.float())).all()), (
        f"m={m}: poisoning every shard row left finite outputs, so the shared "
        "addend does not reach the epilogue"
    )


@pytest.mark.parametrize("m", M_VALUES)
def test_mailbox_rows_past_m_keep_the_lamport_sentinel(m):
    """The write side of the same bound.

    ``lamport_copy`` resets exactly the ``m * H`` elements it consumed. If the
    up-projection epilogue stored past row ``m``, those words stay
    non-sentinel forever, and the next step at a larger ``m`` sees them as
    already-arrived peer data and skips the spin -- reading the previous
    step's values into a live token.
    """
    rank, device = _setup()
    op = _op(device)
    latent = _uniform((MAX_M, L), 4242, device)
    up_w = _uniform((H, L), 4243, device, scale=0.02)
    live = _uniform((m, H), 6000 + m, device)

    _, mailbox = _run_up_projection(op, device, rank, m, live, 0.0, latent, up_w)
    _sync(device)

    words = mailbox.view(torch.int32)[0]
    dirty = words != LAMPORT_EMPTY_I32
    dirty_rows = sorted(set(dirty.nonzero()[:, 0].tolist()))
    assert int(dirty.sum()) == 0, (
        f"m={m}: after the gather, {int(dirty.sum())} mailbox words in rows "
        f"{dirty_rows} are not the Lamport empty sentinel. Rows >= {m} were "
        "written by an epilogue that ignored the live row bound; rows < "
        f"{m} were not reset by the gather."
    )


@pytest.mark.parametrize("m", M_VALUES)
def test_full_tail_op_ignores_a_poisoned_reduce_scatter_tail(m):
    """End-to-end through ``KimiK3LatentTailOp.__call__``.

    Poisoning ``CollectiveKernel._shared_output`` before the call reproduces
    the real condition exactly: the collective overwrites rows ``[0, m)`` and
    leaves rows ``[m, max_m)`` holding whatever was there before.
    """
    rank, device = _setup()
    op = _op(device)
    routed = _uniform((m, L), 7000 + m, device)
    shared = _uniform((m, H), 7100 + m, device)
    rms_w = _uniform((L,), 7200, device, scale=1.0)
    up_w = _uniform((H, L), 7300, device, scale=0.02)
    shared_out = op._collective._shared_output

    shared_out.zero_()
    base = op(routed, shared, rms_w, up_w).clone()
    _sync(device)

    for fill in (float("nan"), 3.0e4):
        shared_out.fill_(fill)
        poisoned = op(routed, shared, rms_w, up_w)
        torch.cuda.synchronize()
        assert torch.equal(_bits(base), _bits(poisoned)), (
            f"m={m}, tail poison {fill}: the fused tail's output depends on "
            f"_shared_output rows [{m}, {MAX_M}), which the reduce-scatter "
            "never writes at this token count"
        )
        assert torch.isfinite(
            poisoned.float()
        ).all(), f"m={m}: poison {fill} propagated into the live output"
        _sync(device)


def test_graph_replays_at_mixed_sizes_stay_isolated():
    """One graph per captured size, replayed in an adversarial order.

    The runtime captures a graph per CUDA-graph batch size against one shared
    ``KimiK3LatentTailOp`` instance, so an ``m=16`` replay leaves real
    shared-expert output in ``_shared_output`` rows ``0..15`` and a
    subsequent ``m=1`` replay -- whose tile bounds were frozen at capture --
    runs straight over it. Poison is also written between replays of the same
    graph so a stale read is unmistakable.
    """
    rank, device = _setup()
    op = _op(device)
    rms_w = _uniform((L,), 8200, device, scale=1.0)
    up_w = _uniform((H, L), 8300, device, scale=0.02)
    shared_out = op._collective._shared_output

    inputs, graphs, outs = {}, {}, {}
    for m in GRAPH_SIZES:
        inputs[m] = (
            _uniform((m, L), 8000 + m, device),
            _uniform((m, H), 8100 + m, device),
        )
        for _ in range(3):  # warm the JIT and the Lamport buffer rotation
            op(*inputs[m], rms_w, up_w)
        _sync(device)
    for m in GRAPH_SIZES:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs[m] = op(*inputs[m], rms_w, up_w)
        graphs[m] = graph
        _sync(device)

    clean = {}
    for m in GRAPH_SIZES:
        shared_out.zero_()
        graphs[m].replay()
        torch.cuda.synchronize()
        clean[m] = outs[m].clone()
        _sync(device)

    for m in GRAPH_SIZES:
        graphs[MAX_M].replay()
        torch.cuda.synchronize()
        graphs[m].replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(clean[m]), _bits(outs[m])), (
            f"replaying the m={MAX_M} graph before the m={m} graph changed the "
            f"m={m} result: rows [{m}, {MAX_M}) of the previous step leaked in"
        )
        _sync(device)

        shared_out.fill_(float("nan"))
        graphs[m].replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(clean[m]), _bits(outs[m])), (
            f"NaN poison in _shared_output rows [{m}, {MAX_M}) changed the "
            f"m={m} graph replay's output"
        )
        assert torch.isfinite(outs[m].float()).all()
        _sync(device)

    # The reverse order exercises the mailbox self-reset: a small replay must
    # not leave state that a following full-width replay consumes.
    shared_out.zero_()
    graphs[1].replay()
    torch.cuda.synchronize()
    graphs[MAX_M].replay()
    torch.cuda.synchronize()
    assert torch.equal(
        _bits(clean[MAX_M]), _bits(outs[MAX_M])
    ), f"an m=1 replay perturbed the following m={MAX_M} replay"
    _sync(device)
