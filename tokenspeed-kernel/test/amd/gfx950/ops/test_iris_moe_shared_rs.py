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

"""Verify the K3 decode MoE two-shot that only reduce-scatters the shared rows.

Every routed element and every element of this rank's shared column block
must keep the full two-shot all-reduce's bits, and so must the gathered tail,
at decode rows through ``K3_FUSED_DECODE_MAX_ROWS``.
"""

import re
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import ArchVersion, current_platform
from utils import assert_no_triton_compile, votes_with_subgroup_mask

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd, reason="Iris requires AMD ROCm"
)

_ROUTED = 3584
_SHARED = 7168
_BLOCK = _SHARED // 8
_TOPK = 16
_EPS = 1e-6


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


@pytest.mark.parametrize("topk", (0, _TOPK))
def test_moe_shared_rs_codegen(topk, tmp_path):
    """Both program kinds compile for gfx950 without spills."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.moe_reduce import (
        iris_k3moe_lamport_twoshot_shared_rs,
    )

    fn = iris_k3moe_lamport_twoshot_shared_rs
    constants = {
        "RANK": 5,
        "WORLD_SIZE": 8,
        "ROUTED_WIDTH": _ROUTED,
        "SHARED_WIDTH": _SHARED,
        "TOPK": topk,
        "MAX_SHARD_ELEMENTS": 192 * (_ROUTED + _SHARED) // 8,
        "NUM_STAGES": 3,
        "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
            ArchVersion(9, 5)
        ).peer_signal_store,
    }
    signature = {}
    attrs = {}
    for index, name in enumerate(fn.arg_names):
        if name in constants:
            continue
        if name == "regions":
            signature[name] = ("i64",) * 8
            attrs.update(
                {(index, peer): [["tt.divisibility", 16]] for peer in range(8)}
            )
            continue
        if name == "ROUTED_ELEMENTS":
            signature[name] = "i32"
            continue
        signature[name] = "*i32" if name == "epochs" else "*bf16"
        attrs[(index,)] = [["tt.divisibility", 16]]
    source = gluon._runtime.GluonASTSource(
        fn, signature, constexprs=constants, attrs=attrs
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 1},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"moe-shared-rs-topk{topk}.amdgcn").write_text(assembly)
    assert re.search(r"\.vgpr_spill_count:\s+0\b", assembly)
    assert re.search(r"\.sgpr_spill_count:\s+0\b", assembly)
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)
    # Group system-scope peer loads by the loop that LLVM reports for them.
    loops = {}
    header = None
    for line in assembly.splitlines():
        if match := re.match(r"^\.L(BB\w+):", line):
            loop = re.search(r"Header=(BB\w+)|(=>This Inner Loop Header)", line)
            header = (loop[1] or match[1]) if loop else None
        elif header:
            loops.setdefault(header, []).append(line)
    polls = [
        body
        for body in loops.values()
        if any("buffer_load_dword " in line and "sc0 sc1" in line for line in body)
    ]
    # The routed programs' two exchanges and the shared programs' one.
    assert sorted(
        sum("buffer_load_dword " in line and "sc0 sc1" in line for line in body)
        for body in polls
    ) == [7, 7, 7], "each exchange must poll all seven peers per retry"
    assert all(votes_with_subgroup_mask(body) for body in polls)
    stores = re.findall(r"^\s*(buffer_store_\w+)[^\n]*\bsc0 sc1\b", assembly, re.M)
    assert stores and set(stores) == {"buffer_store_dword"}, sorted(set(stores))


def _shared_rs_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=300),
    )
    try:
        group = dist.new_group(backend="nccl", timeout=timedelta(seconds=300))
        dist.barrier(group=group, device_ids=[rank])
        _check_shared_rs(rank, device, group)
    finally:
        dist.destroy_process_group()


def _check_shared_rs(rank: int, device: torch.device, group: dist.ProcessGroup) -> None:
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        K3_FUSED_DECODE_MAX_ROWS,
        prepare_k3_communication,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.moe_reduce import (
        iris_k3moe_lamport_twoshot_shared_rs,
    )

    max_rows = K3_FUSED_DECODE_MAX_ROWS
    state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=max_rows * (_ROUTED + _SHARED),
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=True,
        sharded_max_rows=0,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    producer = state.all_reduce
    owned = slice(rank * _BLOCK, (rank + 1) * _BLOCK)
    generator = torch.Generator(device=device).manual_seed(20261110)
    weight = (
        torch.randn((_SHARED, _ROUTED), generator=generator, device=device) / 60
    ).to(torch.bfloat16)
    norm_weight = (torch.rand((_ROUTED,), generator=generator, device=device) + 0.5).to(
        torch.bfloat16
    )
    inputs = {}

    def partials(rows, seed):
        if seed not in inputs:
            source = torch.Generator(device=device).manual_seed(seed * 8 + rank)
            slots = torch.randn(
                (rows * _TOPK, _ROUTED), generator=source, device=device
            ).to(torch.bfloat16)
            routed = torch.randn((rows, _ROUTED), generator=source, device=device).to(
                torch.bfloat16
            )
            shared = (
                torch.randn((rows, _SHARED), generator=source, device=device) * 0.1
            ).to(torch.bfloat16)
            # Negative zeros reach the mailbox only as sanitized zeros.
            for tensor in (slots, routed, shared):
                tensor[:, :8] = -0.0
                tensor[:, _BLOCK - 4 : _BLOCK + 4] = -0.0
            # Cancellation sums give zeros of either sign before sanitizing.
            shared[:, 16:24] = 1.0 if rank % 2 else -1.0
            # The residual is replicated: every rank draws the same rows.
            replicated = torch.Generator(device=device).manual_seed(seed)
            prefix = torch.randn(
                (rows, _SHARED), generator=replicated, device=device
            ).to(torch.bfloat16)
            inputs[seed] = slots, routed, shared, prefix
        return inputs[seed]

    def reduce(rows, seed, topk, shared_owned_columns):
        slots, routed, shared, _ = partials(rows, seed)
        outputs = producer.acquire_outputs(((rows, _ROUTED), (rows, _SHARED)))
        if topk:
            outputs[0].fill_(float("nan"))
        else:
            outputs[0].copy_(routed)
        outputs[1].copy_(shared)
        return state.reduce_moe(
            outputs,
            routed_partials=slots if topk else None,
            shared_owned_columns=shared_owned_columns,
        )

    def tail(rows, seed, reduced):
        prefix = partials(rows, seed)[3]
        assert state.can_moe_gather_tail(*reduced, prefix, weight, norm_weight)
        return state.moe_gather_tail(
            *reduced, prefix, weight, norm_weight=norm_weight, eps=_EPS
        )

    def attention(rows, seed):
        """A plain two-shot all-reduce over the same mailboxes and epochs."""
        source = torch.Generator(device=device).manual_seed(seed * 8 + rank + 7)
        partial = torch.randn((rows, _SHARED), generator=source, device=device).to(
            torch.bfloat16
        )
        return state.reduce_attention(partial)

    def check(rows, seed, topk):
        old_routed, old_shared = (
            t.clone() for t in reduce(rows, seed, topk, shared_owned_columns=False)
        )
        old_tail = tail(rows, seed, (old_routed, old_shared)).clone()
        new_routed, new_shared = reduce(rows, seed, topk, shared_owned_columns=True)
        assert torch.equal(
            new_routed.view(torch.int16), old_routed.view(torch.int16)
        ), f"routed rows={rows} topk={topk}"
        assert torch.equal(
            new_shared[:, owned].view(torch.int16),
            old_shared[:, owned].view(torch.int16),
        ), f"shared rows={rows} topk={topk}"
        new_tail = tail(rows, seed, (new_routed, new_shared))
        assert torch.equal(
            new_tail.view(torch.int16), old_tail.view(torch.int16)
        ), f"tail rows={rows} topk={topk}"

    decode_rows = [r for r in (16, 32, 64, 128, 192) if r <= max_rows]
    # Odd shapes inside the window, the one-shot fallback, and the window's end.
    extra_rows = [2, 4, 7, 8, 10, 48, 56, max_rows - 2, max_rows]
    seed = 0
    for rows in decode_rows + extra_rows + decode_rows[::-1]:
        for topk in (True, False):
            check(rows, seed, topk)
            # Interleave the plain two-shot on the shared mailboxes.
            assert attention(rows, seed) is not None
            seed += 1
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])

    # Captured decode layers, one rank late, without recompiling.
    layer_rows = decode_rows + [8, 4, max_rows] + decode_rows[::-1]
    layers = [
        (rows, 1000 + layer, layer % 2 == 0) for layer, rows in enumerate(layer_rows)
    ]
    expected = []
    for rows, layer_seed, topk in layers:
        reduced = [t.clone() for t in reduce(rows, layer_seed, topk, False)]
        expected.append((reduced, tail(rows, layer_seed, reduced).clone()))
        # Warm the reduce-scatter at this shape before capture.
        tail(rows, layer_seed, reduce(rows, layer_seed, topk, True))
    snapshots = [
        (torch.empty_like(r), torch.empty_like(s[:, owned]), torch.empty_like(t))
        for (r, s), t in expected
    ]
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), assert_no_triton_compile(
        iris_k3moe_lamport_twoshot_shared_rs
    ):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for layer, (rows, layer_seed, topk) in enumerate(layers):
                if rank == (layer * 3 + 1) % 8:
                    torch.cuda._sleep(100_000)
                routed, shared = reduce(rows, layer_seed, topk, True)
                snapshots[layer][0].copy_(routed)
                snapshots[layer][1].copy_(shared[:, owned])
                snapshots[layer][2].copy_(tail(rows, layer_seed, (routed, shared)))
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for (routed, shared, out), ((want_routed, want_shared), want_out) in zip(
            snapshots, expected, strict=True
        ):
            assert torch.equal(routed, want_routed)
            assert torch.equal(shared, want_shared[:, owned])
            assert torch.equal(out, want_out)
    # Ranks hold identical gathered outputs.
    replicas = [torch.empty_like(snapshots[-1][2]) for _ in range(8)]
    dist.all_gather(replicas, snapshots[-1][2].contiguous(), group=group)
    for replica in replicas:
        assert torch.equal(replica, snapshots[-1][2])
    dist.barrier(group=group, device_ids=[rank])


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="the K3 Lamport MoE two-shot runs across eight CDNA4 GPUs",
)
def test_iris_moe_shared_rs_bitwise() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_shared_rs_worker, args=(port,), nprocs=8, join=True)
