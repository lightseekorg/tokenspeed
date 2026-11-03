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

"""Muse Spark fused decoder layers against the regular layer, on 8 gfx950 GPUs.

Each rank builds one random-weight layer per plan variant through the shared
harness and compares the fused launch with today's regular Muse ops on the
same inputs (routing untied, so both pick the same experts). The launch state
is shared by every launch of a group, so the same state then serves
alternating request counts with a regular collective between launches, a
captured graph whose replay must equal eager and whose topology must be a
chain (no node can run beside a fused launch).
"""

import ctypes
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
if not (
    torch.cuda.is_available()
    and torch.cuda.device_count() >= WORLD
    and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
):
    pytest.skip("needs 8 gfx950 GPUs", allow_module_level=True)

harness = pytest.importorskip("tokenspeed_kernel.ops.decoder_layer.testing")

from tokenspeed_kernel.ops.decoder_layer import LAYER_KINDS, PlanKey  # noqa: E402

PLAN_KEYS = [
    PlanKey(requests, kind, next_kind)
    for requests in (1, 2)
    for kind in LAYER_KINDS
    for next_kind in LAYER_KINDS + ("regular",)
]


def _run(fused, key, inputs):
    """One fused launch on ``inputs``: (next query or normed rows, residual)."""
    common = dict(positions=inputs["positions"], cache=inputs["cache"])
    if key.next_kind == "regular":
        return fused.forward_last(
            inputs["prepared_query"],
            inputs["residual"].clone(),
            next_norm=inputs["next_norm"],
            **common,
        )
    return fused.forward(
        inputs["prepared_query"],
        inputs["residual"].clone(),
        next_prologue=inputs["next_prologue"],
        **common,
    )


def _assert_close_to_regular(key, actual, expected):
    head, residual = actual
    expected_head, expected_residual = expected
    # FP32 residual after BF16 GEMV outputs: one BF16 ULP (2**-8 relative)
    # per rounding, two roundings on the path (rtol 2**-7); atol for sums
    # near zero, far below a BF16 ULP of the |x| ~ 1 rows.
    torch.testing.assert_close(residual, expected_residual, atol=2**-10, rtol=2**-7)
    if key.next_kind == "regular":
        # Normed BF16 rows: the residual's error plus one BF16 rounding.
        torch.testing.assert_close(
            head.float(), expected_head.float(), atol=2**-9, rtol=2**-6
        )
        return
    # The next query: FP8 E4M3 q (3 mantissa bits, one ULP = 2**-3
    # relative) and the BF16 output gate.
    (q, gate), (expected_q, expected_gate) = head, expected_head
    torch.testing.assert_close(q.float(), expected_q.float(), atol=2**-6, rtol=2**-3)
    torch.testing.assert_close(
        gate.float(), expected_gate.float(), atol=2**-9, rtol=2**-6
    )


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return tuple(_clone(item) for item in value)


def _hip():
    bundled = os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so")
    return ctypes.CDLL(bundled if os.path.exists(bundled) else "libamdhip64.so")


def _graph_is_chain(graph: torch.cuda.CUDAGraph) -> bool:
    """Every node of the captured graph has at most one predecessor and one
    successor, so no node can run concurrently with another."""
    hip, raw = _hip(), ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t(0)
    assert hip.hipGraphGetEdges(raw, None, None, ctypes.byref(count)) == 0
    sources = (ctypes.c_void_p * count.value)()
    targets = (ctypes.c_void_p * count.value)()
    assert hip.hipGraphGetEdges(raw, sources, targets, ctypes.byref(count)) == 0
    outgoing, incoming = {}, {}
    for source, target in zip(sources, targets):
        outgoing[source] = outgoing.get(source, 0) + 1
        incoming[target] = incoming.get(target, 0) + 1
    return (
        max(outgoing.values(), default=0) <= 1
        and max(incoming.values(), default=0) <= 1
    )


def _worker(rank, init_file, failures):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=120),
    )
    device = torch.device(f"cuda:{rank}")
    try:
        built = {
            key: harness.build_random_layer(
                recipe="muse_spark",
                plan_key=key,
                rank=rank,
                world=WORLD,
                seed=0,
                device=device,
            )
            for key in PLAN_KEYS
        }
        references = {
            key: harness.regular_layer_reference(weights, inputs)
            for key, (_, weights, inputs) in built.items()
        }

        # Every plan variant against the regular layer.
        firsts = {}
        for key, (fused, _, inputs) in built.items():
            actual = _run(fused, key, inputs)
            _assert_close_to_regular(key, actual, references[key])
            # Outputs may live in buffers the next launch reuses.
            firsts[key] = _clone(actual)

        # The residual is replicated: every rank must hold the same bits.
        for key, (_, residual) in firsts.items():
            gathered = [torch.empty_like(residual) for _ in range(WORLD)]
            dist.all_gather(gathered, residual)
            assert all(torch.equal(other, residual) for other in gathered), key

        # One state across launches: alternate 1 and 2 requests (the
        # 2-request launch leaves rows the 1-request one must not accept)
        # with a regular collective between; every launch repeats its first
        # result bit for bit.
        sliding = [key for key in PLAN_KEYS if key.layer_kind == "sliding"]
        for key in sorted(sliding, key=lambda key: key.requests) * 3:
            fused, _, inputs = built[key]
            head, residual = _run(fused, key, inputs)
            dist.all_reduce(torch.ones(1, device=device))
            assert torch.equal(residual, firsts[key][1]), key

        # Replay equals eager, and the captured graph is a chain.
        key = PlanKey(1, "sliding", "sliding")
        fused, _, inputs = built[key]
        static = dict(inputs, residual=inputs["residual"].clone())
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            captured = _run(fused, key, static)
        assert _graph_is_chain(graph)
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
        (q, gate), residual = captured
        (eager_q, eager_gate), eager_residual = firsts[key]
        assert torch.equal(residual, eager_residual)
        assert torch.equal(q.view(torch.uint8), eager_q.view(torch.uint8))
        assert torch.equal(gate, eager_gate)
    except BaseException as error:  # noqa: BLE001 - reported to the parent
        failures[rank] = repr(error)
        raise
    finally:
        dist.destroy_process_group()


def test_fused_layers_match_the_regular_layer_and_reuse_one_state(tmp_path):
    with mp.Manager() as manager:
        failures = manager.dict()
        try:
            mp.spawn(
                _worker,
                args=(str(tmp_path / "rendezvous"), failures),
                nprocs=WORLD,
                join=True,
            )
        finally:
            failures = dict(failures)
    assert failures == {}
