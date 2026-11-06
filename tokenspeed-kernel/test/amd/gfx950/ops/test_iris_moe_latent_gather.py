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

"""Verify the K3 decode MoE input projection with sharded replicated columns."""

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

_HIDDEN = 7168
_ROUTER = 896
_LATENT = 3584
_SHARED = 768
_PACKED = _ROUTER + _LATENT + 2 * _SHARED


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


@pytest.mark.parametrize("quantize", (False, True))
@pytest.mark.parametrize("rank", (0, 7))
def test_latent_input_gather_codegen(rank, quantize, tmp_path):
    """Publishers store whole words; the gathering retry rereads its words and
    votes with one lane mask."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.latent_gather import (
        iris_k3moe_latent_input_gather,
    )

    fn = iris_k3moe_latent_input_gather
    constants = {
        "RANK": rank,
        "WORLD_SIZE": 8,
        "SPLIT_K": 16,
        "TOTAL_N": _PACKED,
        "ROUTER_N": _ROUTER,
        "LATENT_N": _LATENT,
        "SHARED_N": _SHARED,
        "HAS_LINEAR_BETA": True,
        "MAX_ELEMENTS": 6 * (_LATENT + _HIDDEN),
        "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
            ArchVersion(9, 5)
        ).peer_signal_store,
        "NUM_WARPS": 4,
        "QUANTIZE": quantize,
        "MIN_AMAX": 1.0e-12,
        "FP8_MAX": 448.0,
        "MAX_ROWS": 64,
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
        if name in ("beta", "linear_beta"):
            signature[name] = "fp32"
            continue
        if name in ("tokens", "split_stride"):
            signature[name] = "i32"
            continue
        signature[name] = {
            "partial_ptr": "*fp32",
            "router_ptr": "*fp32",
            "fp8_ptr": "*fp8e4nv",
            "sync_ptr": "*i32",
            "scale_ptr": "*fp32",
            "epochs": "*i32",
        }.get(name, "*bf16")
        attrs[(index,)] = [["tt.divisibility", 16]]
    kernel = triton.compile(
        gluon._runtime.GluonASTSource(fn, signature, constexprs=constants, attrs=attrs),
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 4},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"latent-gather-rank{rank}-{quantize}.amdgcn").write_text(assembly)
    assert not re.search(r"\bbuffer_(?:load_ushort|store_short)\b", assembly)
    assert re.search(r"\.vgpr_spill_count:\s+0\b", assembly)
    polls = [
        cycle
        for cycle in polling_cycles(assembly)
        if sum("buffer_load_dwordx4" in line and "sc0 sc1" in line for line in cycle)
        == 4
    ]
    assert polls, "all four system-cache word reads must execute again on a retry"
    assert all(votes_with_subgroup_mask(cycle) for cycle in polls)


def _latent_gather_worker(rank: int, port: int) -> None:
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
        _check_latent_gather(rank, device, group)
    finally:
        dist.destroy_process_group()


def _check_latent_gather(
    rank: int, device: torch.device, group: dist.ProcessGroup
) -> None:
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.latent_gather import (
        iris_k3moe_latent_input_gather,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_small_batch import (
        launch_gluon_latent_input_small_batch_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.quantize import (
        _dynamic_fp8_quantize,
    )

    state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=64 * (_LATENT + _HIDDEN),
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=True,
        sharded_max_rows=0,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    # Router and routed rows are replicated; shared rows are each rank's own shard.
    generator = torch.Generator(device=device).manual_seed(20261030)
    weight = (
        torch.randn((_PACKED, _HIDDEN), generator=generator, device=device) / 40
    ).to(torch.bfloat16)
    weight[_ROUTER + _LATENT :] = (
        torch.randn((2 * _SHARED, _HIDDEN), device=device) / 40
    ).to(torch.bfloat16)
    beta, linear_beta = 25.0, 25.0

    def replicated(hidden, quantize=False):
        router, routed, shared = launch_gluon_latent_input_small_batch_gfx950(
            hidden,
            weight[:_ROUTER],
            weight[_ROUTER : _ROUTER + _LATENT],
            weight[_ROUTER + _LATENT :],
            weight,
            beta=beta,
            linear_beta=linear_beta,
        )
        # The mailbox carries negative zeros as zeros.
        routed = torch.where(routed == 0, 0.0, routed).to(routed.dtype)
        result = (torch.where(router == 0, 0.0, router), routed, shared)
        if quantize:
            # The MoE's own quantizer of the gathered latent.
            result += _dynamic_fp8_quantize(routed)
        return result

    def sharded(hidden, quantize=False):
        assert state.can_latent_input_gather(hidden, weight)
        result = state.latent_input_gather(
            hidden, weight, beta=beta, linear_beta=linear_beta, quantize=quantize
        )
        if quantize:
            *result, (routed_fp8, routed_scale) = result
            result = (*result, routed_fp8, routed_scale)
        return result

    def bits(tensor):
        if tensor.dtype == torch.float8_e4m3fn:
            return tensor.view(torch.uint8)
        return tensor.view(
            torch.int16 if tensor.dtype != torch.float32 else torch.int32
        )

    def same(got, want):
        return all(
            torch.equal(bits(a), bits(b)) for a, b in zip(got, want, strict=True)
        )

    hidden_rows = {}

    def hidden_for(rows, seed):
        if seed not in hidden_rows:
            source = torch.Generator(device=device).manual_seed(seed)
            hidden_rows[seed] = torch.randn(
                (rows, _HIDDEN), generator=source, device=device
            ).to(torch.bfloat16)
        return hidden_rows[seed]

    # Interleave with the MoE all-reduce, which shares the mailbox tiles.
    producer = state.all_reduce
    for seed, rows in enumerate((4, 2, 64, 6, 16, 17, 3, 8, 32, 5, 4, 64, 32)):
        hidden = hidden_for(rows, seed)
        quantize = seed % 3 != 1
        assert same(sharded(hidden, quantize), replicated(hidden, quantize))
        reduce_rows = min(rows, 6)
        outputs = producer.acquire_outputs(
            ((reduce_rows, _LATENT), (reduce_rows, _HIDDEN))
        )
        for output in outputs:
            output.fill_(1.0)
        reduced = state.reduce_moe(outputs)
        assert torch.equal(reduced[0], torch.full_like(reduced[0], 8.0))

    single = hidden_for(1, 100)
    assert not state.can_latent_input_gather(single, weight)
    assert not state.can_latent_input_gather(hidden_for(65, 101), weight)
    assert not state.can_latent_input_gather(hidden_for(4, 4).float(), weight)
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])

    # Captured decode layers with one rank late, at row counts sharing a binary.
    layers = 6
    expected = [
        replicated(hidden_for(4 - layer % 2, 200 + layer), layer % 3 != 2)
        for layer in range(layers)
    ]
    for layer in range(layers):
        sharded(hidden_for(4 - layer % 2, 200 + layer), layer % 3 != 2)
    snapshots = [[torch.empty_like(t) for t in layer] for layer in expected]
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), assert_no_triton_compile(
        iris_k3moe_latent_input_gather
    ):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for layer in range(layers):
                if rank == (layer * 3 + 1) % 8:
                    torch.cuda._sleep(100_000)
                for snapshot, result in zip(
                    snapshots[layer],
                    sharded(hidden_for(4 - layer % 2, 200 + layer), layer % 3 != 2),
                    strict=True,
                ):
                    snapshot.copy_(result)
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for layer_snapshots, layer_expected in zip(snapshots, expected, strict=True):
            assert same(layer_snapshots, layer_expected)
    dist.barrier(group=group, device_ids=[rank])


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="the K3 latent-input gather runs across eight CDNA4 GPUs",
)
def test_iris_moe_latent_input_gather_bitwise() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_latent_gather_worker, args=(port,), nprocs=8, join=True)
