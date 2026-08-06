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

"""Staged NVLS multimem all-reduce: must match NCCL all-reduce across the
latent (3584) and hidden (7168) widths on both sides of every runtime
threshold, survive buffer growth (doubling re-allocation + re-rendezvous,
replaced buffers retired not freed, no shrink on smaller re-stage), keep the
per-width buffer caches independent, refuse unsupported inputs by returning
None, and honor the documented view-invalidation contract.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

LATENT, HIDDEN = 3584, 7168
# Spans the runtime MULTIMEM_AR window floor (17) and _MIN_BUFFER_ROWS
# (2048) in the staging module, on both sides of each.
M_VALUES = [17, 64, 1024, 2047, 2048, 4096]


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


def _require_multimem():
    # Collectively-agreed probe (all_reduce inside): every rank calls it in
    # lockstep at the top of every test, so skips agree across ranks.
    from tokenspeed_kernel.ops.communication.multimem import (
        multimem_available_all_ranks,
    )

    if not multimem_available_all_ranks():
        pytest.skip("platform has no rank-agreed multicast (NVLS) support")


def _group_name() -> str:
    # Mirrors the runtime call site in kimi_k3._tail_multimem_ar.
    return dist.group.WORLD.group_name


def _inputs(rank: int, dev: torch.device, m: int, width: int, seed: int):
    torch.manual_seed(seed * 7919 + rank)  # per-rank partials
    return (torch.randn(m, width, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()


def _nccl_reference(x: torch.Tensor) -> torch.Tensor:
    ref = x.clone()
    dist.all_reduce(ref)
    return ref


def _staged_reduce(x: torch.Tensor) -> torch.Tensor:
    from tokenspeed_kernel.ops.communication.multimem import (
        multimem_all_reduce_staged,
        multimem_stage,
    )

    view = multimem_stage(x, _group_name())
    assert view is not None, "multimem_stage refused a supported input"
    assert view.shape == x.shape
    return multimem_all_reduce_staged(view, _group_name())


def _assert_matches(out: torch.Tensor, ref: torch.Tensor, ctx: str):
    torch.cuda.synchronize()
    scale = ref.float().abs().max().item()
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.05 * max(scale, 1.0), f"{ctx}: err {err} vs scale {scale}"


@pytest.mark.parametrize("width", [LATENT, HIDDEN])
@pytest.mark.parametrize("m", M_VALUES)
def test_staged_ar_matches_nccl(m, width):
    rank, dev = _setup()
    _require_multimem()
    x = _inputs(rank, dev, m, width, seed=100 + m)
    ref = _nccl_reference(x)
    out = _staged_reduce(x)
    _assert_matches(out, ref, f"m={m} width={width}")


def test_buffer_growth_reallocates_and_stays_correct():
    """m=64 (fresh min-capacity buffer), m=2049 (grows by doubling, not to
    rows), m=4096 (exactly fits the doubled buffer — no re-allocation), then
    m=64 again (reuses the grown buffer, no shrink) — all correct, and the
    replaced buffer is retired, not freed."""
    from tokenspeed_kernel.ops.communication import multimem

    rank, dev = _setup()
    _require_multimem()
    # Retire any cached buffers for this width on every rank symmetrically so
    # a fresh allocation happens regardless of what earlier tests staged
    # (retire rather than free: peers may still hold multicast mappings).
    torch.cuda.synchronize()
    dist.barrier()
    for key in [k for k in multimem._BUFFERS if k[1] == LATENT]:
        multimem._RETIRED.append(multimem._BUFFERS.pop(key))
    dist.barrier()
    key = (dev.index, LATENT, _group_name())
    # step -> (m, expected capacity, expect re-allocation)
    plan = [
        (64, 2048, True),
        (2049, 4096, True),
        (4096, 4096, False),
        (64, 4096, False),
    ]
    retired_before = len(multimem._RETIRED)
    prev_buf = None
    for step, (m, expected_cap, expect_realloc) in enumerate(plan):
        x = _inputs(rank, dev, m, LATENT, seed=500 + step)
        ref = _nccl_reference(x)
        out = _staged_reduce(x)
        _assert_matches(out, ref, f"growth step {step} m={m}")
        buf = multimem._BUFFERS[key]
        assert (
            buf.shape[0] == expected_cap
        ), f"step {step} m={m}: capacity {buf.shape[0]} != {expected_cap}"
        if prev_buf is not None:
            realloced = buf.data_ptr() != prev_buf.data_ptr()
            assert (
                realloced == expect_realloc
            ), f"step {step} m={m}: realloc={realloced}, expected {expect_realloc}"
        prev_buf = buf
    # Exactly one growth after the fresh allocation -> exactly one retirement.
    assert (
        len(multimem._RETIRED) == retired_before + 1
    ), "replaced buffer must be kept alive on the retired list"


def test_width_caches_are_independent():
    """Interleaved 3584- and 7168-wide reduces must not cross-contaminate:
    staging one width leaves the other width's staged data intact."""
    from tokenspeed_kernel.ops.communication.multimem import (
        multimem_all_reduce_staged,
        multimem_stage,
    )

    rank, dev = _setup()
    _require_multimem()
    m = 64
    a = _inputs(rank, dev, m, LATENT, seed=800)
    b = _inputs(rank, dev, m, HIDDEN, seed=801)
    ref_a = _nccl_reference(a)
    ref_b = _nccl_reference(b)
    # Stage A, then stage B (other width), then reduce both: if the caches
    # shared storage, staging B would corrupt A's staged bytes.
    view_a = multimem_stage(a, _group_name())
    view_b = multimem_stage(b, _group_name())
    assert view_a is not None and view_b is not None
    assert view_a.data_ptr() != view_b.data_ptr()
    out_a = multimem_all_reduce_staged(view_a, _group_name())
    out_b = multimem_all_reduce_staged(view_b, _group_name())
    _assert_matches(out_a, ref_a, "interleaved width 3584")
    _assert_matches(out_b, ref_b, "interleaved width 7168")
    # And in the opposite interleaving order.
    c = _inputs(rank, dev, m, HIDDEN, seed=802)
    d = _inputs(rank, dev, m, LATENT, seed=803)
    ref_c = _nccl_reference(c)
    ref_d = _nccl_reference(d)
    view_c = multimem_stage(c, _group_name())
    view_d = multimem_stage(d, _group_name())
    out_c = multimem_all_reduce_staged(view_c, _group_name())
    out_d = multimem_all_reduce_staged(view_d, _group_name())
    _assert_matches(out_c, ref_c, "interleaved width 7168 (reversed)")
    _assert_matches(out_d, ref_d, "interleaved width 3584 (reversed)")


def test_stage_refuses_unsupported_tensors():
    """multimem_stage must return None (fallback signal) for CPU, non-bf16,
    non-2-D, and misaligned-width tensors, without raising."""
    from tokenspeed_kernel.ops.communication.multimem import multimem_stage

    _, dev = _setup()
    _require_multimem()  # None-returns below must come from validation, not
    # from multimem being unavailable altogether.
    name = _group_name()
    cpu = torch.randn(4, LATENT, dtype=torch.bfloat16, device="cpu")
    assert multimem_stage(cpu, name) is None, "CPU tensor must be refused"
    fp32 = torch.randn(4, LATENT, dtype=torch.float32, device=dev)
    assert multimem_stage(fp32, name) is None, "fp32 tensor must be refused"
    three_d = torch.randn(2, 4, LATENT, dtype=torch.bfloat16, device=dev)
    assert multimem_stage(three_d, name) is None, "3-D tensor must be refused"
    misaligned = torch.randn(4, LATENT + 1, dtype=torch.bfloat16, device=dev)
    assert multimem_stage(misaligned, name) is None, "width % 8 must be refused"


def test_stage_view_invalidated_by_next_stage_of_same_width():
    """The documented contract: a returned view is only valid until the next
    stage of the same width — after that its content is the new payload, so
    callers must never rely on stale views."""
    from tokenspeed_kernel.ops.communication.multimem import multimem_stage

    rank, dev = _setup()
    _require_multimem()
    m = 64
    a = _inputs(rank, dev, m, LATENT, seed=900)
    b = _inputs(rank, dev, m, LATENT, seed=901)
    assert not torch.equal(a, b)
    view_a = multimem_stage(a, _group_name())
    assert view_a is not None
    assert torch.equal(view_a, a)
    view_b = multimem_stage(b, _group_name())
    assert view_b is not None
    assert view_b.data_ptr() == view_a.data_ptr(), "same-width stages share storage"
    torch.cuda.synchronize()
    assert torch.equal(view_a, b), "old view must hold the NEW payload"
    assert not torch.equal(view_a, a), "old view must no longer hold its payload"


def test_graph_replay_survives_buffer_growth():
    """A captured staged reduce must stay valid after the buffer is retired."""
    from tokenspeed_kernel.ops.communication import multimem
    from tokenspeed_kernel.ops.communication.multimem import (
        multimem_all_reduce_staged,
        multimem_stage,
    )

    rank, dev = _setup()
    _require_multimem()
    gname = _group_name()
    # Reset this width symmetrically so the later stage really grows/retires,
    # regardless of what earlier tests left cached.
    torch.cuda.synchronize()
    dist.barrier()
    for key in [k for k in multimem._BUFFERS if k[1] == LATENT]:
        multimem._RETIRED.append(multimem._BUFFERS.pop(key))
    dist.barrier()
    retired_before = len(multimem._RETIRED)
    small = _inputs(rank, dev, 64, 3584, seed=41_000)
    ref_small = _nccl_reference(small)

    # Warm the path eagerly, then capture the m=64 staged reduce.
    view = multimem_stage(small, gname)
    assert view is not None
    multimem_all_reduce_staged(view, gname)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = multimem_stage(small, gname)
        assert captured is not None
        multimem_all_reduce_staged(captured, gname)

    # Eager growth retires the captured buffer's slot in _BUFFERS.
    big = _inputs(rank, dev, 4096, 3584, seed=41_001)
    ref_big = _nccl_reference(big)
    grown = multimem_stage(big, gname)
    assert grown is not None
    assert (
        len(multimem._RETIRED) > retired_before
    ), "growth must retire the captured buffer"
    out_big = multimem_all_reduce_staged(grown, gname)
    torch.cuda.synchronize()
    _assert_matches(out_big, ref_big, "post-growth eager")
    dist.barrier()

    # Replay must still reduce through the retired (kept-alive) buffer.
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    _assert_matches(captured, ref_small, "replay after growth")
    dist.barrier()
