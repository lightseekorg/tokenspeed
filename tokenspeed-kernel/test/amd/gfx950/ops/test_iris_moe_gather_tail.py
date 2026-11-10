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

"""Verify the K3 decode MoE tail that projects column blocks and gathers them."""

import re
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import ArchVersion, current_platform
from utils import assert_no_triton_compile, polling_cycles, votes_with_subgroup_mask

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd, reason="Iris requires AMD ROCm"
)

_LATENT = 3584
_HIDDEN = 7168
_EPS = 1e-6


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


@pytest.mark.parametrize("block_m", (1, 2, 4, None))
def test_moe_gather_tail_codegen(block_m, tmp_path):
    """Senders keep 16-byte vectors in registers; the reducer's retry rereads its
    words and votes with one lane mask. ``None`` is the MFMA sender."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.moetail_gather import (
        K3_GATHER_ROW_STRIDE,
        iris_k3moe_gemv_lamport_gather,
        iris_k3moe_mfma_lamport_gather,
    )

    constants = {
        "RANK": 3,
        "WORLD_SIZE": 8,
        "K": _LATENT,
        "LOCAL_WIDTH": _HIDDEN // 8,
        "MAX_ELEMENTS": 64 * K3_GATHER_ROW_STRIDE,
        "HAS_NORM": True,
        "NUM_WARPS": 8,
        "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
            ArchVersion(9, 5)
        ).peer_signal_store,
    }
    if block_m is None:
        fn = iris_k3moe_mfma_lamport_gather
        constants["CHUNK"] = 64
    else:
        fn = iris_k3moe_gemv_lamport_gather
        constants.update(
            {
                "ROW_STRIDE": K3_GATHER_ROW_STRIDE,
                "MAX_ELEMENTS": 6 * (_LATENT + _HIDDEN),
                "BLOCK_M": block_m,
                "BLOCK_K": 512,
                "WAVES_PER_OUTPUT": 2,
            }
        )
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
        if name == "epochs":
            signature[name] = "*i32"
        elif name in ("M", "latent_stride", "prefix_stride", "shared_stride"):
            # Runtime scalars without divisibility annotations.
            signature[name] = "i32"
            continue
        elif name == "eps":
            signature[name] = "fp32"
            continue
        else:
            signature[name] = "*bf16"
        attrs[(index,)] = [["tt.divisibility", 16]]
    source = gluon._runtime.GluonASTSource(
        fn, signature, constexprs=constants, attrs=attrs
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 8},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"moe-gather-tail-m{block_m}.amdgcn").write_text(assembly)
    assert not re.search(r"\bbuffer_(?:load_ushort|store_short)\b", assembly)
    assert re.search(r"\.vgpr_spill_count:\s+0\b", assembly)
    if block_m is None:
        # Seven K steps of two 16x16x32 MFMAs per subgroup.
        assert len(re.findall(r"\bv_mfma_f32_16x16x32_bf16\b", assembly)) == 14
    polls = [
        cycle
        for cycle in polling_cycles(assembly)
        if sum("buffer_load_dwordx4" in line and "sc0 sc1" in line for line in cycle)
        == 2
    ]
    assert polls, "both system-cache word reads must execute again on a retry"
    assert all(votes_with_subgroup_mask(cycle) for cycle in polls)


def _gather_tail_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    try:
        group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
        dist.barrier(group=group, device_ids=[rank])
        _check_gather_tail(rank, device, group)
    finally:
        dist.destroy_process_group()


def _bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in BF16 representable steps."""

    def ordered(t):
        bits = t.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)

    return (ordered(a) - ordered(b)).abs()


def _check_gather_tail(
    rank: int, device: torch.device, group: dist.ProcessGroup
) -> None:
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.moetail_gather import (
        iris_k3moe_gemv_lamport_gather,
        iris_k3moe_mfma_lamport_gather,
    )
    from tokenspeed_kernel.ops.gemm import kimi3_latent_projection_add3
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

    state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=72 * (_LATENT + _HIDDEN),
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=True,
        sharded_max_rows=0,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    producer = state.all_reduce
    generator = torch.Generator(device=device).manual_seed(20261030)
    weight = (
        torch.randn((_HIDDEN, _LATENT), generator=generator, device=device) / 60
    ).to(torch.bfloat16)
    norm_weight = (torch.rand((_LATENT,), generator=generator, device=device) + 0.5).to(
        torch.bfloat16
    )

    partials = {}

    def reduce(rows, seed):
        """Lamport-reduce this rank's partials, as the MoE producers do."""
        if seed not in partials:
            source = torch.Generator(device=device).manual_seed(seed * 8 + rank)
            partials[seed] = (
                torch.randn((rows, _LATENT), generator=source, device=device).to(
                    torch.bfloat16
                ),
                (
                    torch.randn((rows, _HIDDEN), generator=source, device=device) * 0.1
                ).to(torch.bfloat16),
            )
        inputs = producer.acquire_outputs(((rows, _LATENT), (rows, _HIDDEN)))
        for destination, partial in zip(inputs, partials[seed], strict=True):
            destination.copy_(partial)
        return state.reduce_moe(inputs)

    def ordinary(routed, shared, prefix):
        normed = rmsnorm(routed, norm_weight, _EPS)
        return kimi3_latent_projection_add3(normed, weight, prefix, shared)

    def gathered(routed, shared, prefix):
        assert state.can_moe_gather_tail(routed, shared, prefix, weight, norm_weight)
        return state.moe_gather_tail(
            routed, shared, prefix, weight, norm_weight=norm_weight, eps=_EPS
        )

    # Reductions and gathers share mailbox tile epochs: interleave them at varied rows.
    row_counts = (4, 1, 3, 2, 8, 5, 16, 17, 4, 32, 64, 3, 47)
    for seed, rows in enumerate(row_counts):
        routed, shared = (t.clone() for t in reduce(rows, seed))
        prefix = torch.randn((rows, _HIDDEN), generator=generator, device=device).to(
            torch.bfloat16
        )
        result = gathered(routed, shared, prefix)
        expected = ordinary(routed, shared, prefix)
        exact = (
            prefix.double()
            + shared.double()
            + rmsnorm(routed, norm_weight, _EPS).double() @ weight.double().T
        )
        # At least as accurate as RMSNorm plus the add3 projection.
        error = (result.double() - exact).abs().max()
        assert error <= (expected.double() - exact).abs().max() * 1.01 + 1e-6
        if 1 < rows <= 32:
            # 2-32 rows round once after both adds; only the FP32 sum order differs.
            normed = rmsnorm(routed, norm_weight, _EPS)
            assert state.can_moe_gather_tail(normed, shared, prefix, weight, None)
            projected = state.moe_gather_tail(
                normed, shared, prefix, weight, norm_weight=None, eps=None
            )
            assert (_bf16_ulps(projected, expected) > 0).float().mean() < 0.002
            # The fused norm's sum order can flip one element, perturbing its whole row.
            changed = (result != projected).any(dim=1).sum().item()
            assert changed <= rows // 4 + 1
        replicas = [torch.empty_like(result) for _ in range(8)]
        dist.all_gather(replicas, result, group=group)
        for replica in replicas:
            assert torch.equal(replica, result), "ranks must hold identical rows"
        # Row bits are independent of batch width, GEMV senders (1-4 rows) or MFMA.
        narrow = 1 if rows <= 4 else 5
        assert torch.equal(
            gathered(routed[:narrow], shared[:narrow], prefix[:narrow]),
            result[:narrow],
        )

    # Wider batches and misaligned rows take the ordinary tail.
    routed, shared = reduce(65, 99)
    prefix = torch.zeros((65, _HIDDEN), dtype=torch.bfloat16, device=device)
    assert not state.can_moe_gather_tail(routed, shared, prefix, weight, norm_weight)
    odd_rows = torch.zeros((4, _LATENT + 1), dtype=torch.bfloat16, device=device)[
        :, :_LATENT
    ]
    assert not state.can_moe_gather_tail(
        odd_rows, shared[:4], prefix[:4], weight, norm_weight
    )
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])

    # Captured layers alternate reduce and gather, one rank late, without recompiling.
    layer_rows = (4, 3, 8, 16, 32, 64, 13, 2)
    for rows in layer_rows:
        routed, shared = reduce(rows, 200 + rows)
        gathered(routed, shared, torch.zeros_like(shared))
    prefixes = [
        torch.randn((rows, _HIDDEN), generator=generator, device=device).to(
            torch.bfloat16
        )
        for rows in layer_rows
    ]
    expected = []
    for layer, rows in enumerate(layer_rows):
        routed, shared = (t.clone() for t in reduce(rows, 300 + layer))
        expected.append(gathered(routed, shared, prefixes[layer]).clone())
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    snapshots = [torch.empty_like(t) for t in expected]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), assert_no_triton_compile(
        iris_k3moe_gemv_lamport_gather, iris_k3moe_mfma_lamport_gather
    ):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for layer, rows in enumerate(layer_rows):
                if rank == (layer * 3 + 1) % 8:
                    torch.cuda._sleep(100_000)
                routed, shared = reduce(rows, 300 + layer)
                snapshots[layer].copy_(gathered(routed, shared, prefixes[layer]))
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for snapshot, oracle in zip(snapshots, expected, strict=True):
            assert torch.equal(snapshot, oracle)
    dist.barrier(group=group, device_ids=[rank])


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="the K3 gather tail runs across eight CDNA4 GPUs",
)
def test_iris_moe_gather_tail() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_gather_tail_worker, args=(port,), nprocs=8, join=True)
