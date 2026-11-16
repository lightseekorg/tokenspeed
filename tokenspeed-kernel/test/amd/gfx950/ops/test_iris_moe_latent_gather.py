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


@pytest.mark.parametrize("routed", (False, True))
@pytest.mark.parametrize("quantize", (False, True))
@pytest.mark.parametrize("rank", (0, 7))
def test_latent_input_gather_codegen(rank, quantize, routed, tmp_path):
    """Publishers store whole words; the gathering retry rereads its words and
    votes with one lane mask, also in the routed kernel, which adds the router
    and the shared down projection."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.latent_gather import (
        iris_k3moe_latent_input_gather,
        iris_k3moe_latent_input_gather_routed,
    )

    fn = (
        iris_k3moe_latent_input_gather_routed
        if routed
        else iris_k3moe_latent_input_gather
    )
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
        "MAX_ROWS": 256,
    }
    if routed:
        constants.update(
            ROUTED_SCALING_FACTOR=1.0,
            NORMALIZE_TOPK_WEIGHTS=True,
            BLOCK_SHIFT=4,
            DOWN_N=_HIDDEN,
            DOWN_BLOCK_M=16,
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
        if name in ("beta", "linear_beta"):
            signature[name] = "fp32"
            continue
        if name in ("tokens", "split_stride", "down_out_stride"):
            signature[name] = "i32"
            continue
        signature[name] = {
            "partial_ptr": "*fp32",
            "router_ptr": "*fp32",
            "fp8_ptr": "*fp8e4nv",
            "sync_ptr": "*i32",
            "scale_ptr": "*fp32",
            "epochs": "*i32",
            "bias_ptr": "*fp32",
            "topk_ids": "*i32",
            "topk_weights": "*fp32",
            "sorted_ids": "*i32",
            "sorted_weights": "*fp32",
            "sorted_expert_ids": "*i32",
            "num_valid_ids": "*i32",
            "route_scratch": "*i32",
            "down_sync": "*i32",
        }.get(name, "*bf16")
        attrs[(index,)] = [["tt.divisibility", 16]]
    kernel = triton.compile(
        gluon._runtime.GluonASTSource(fn, signature, constexprs=constants, attrs=attrs),
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 4},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"latent-gather-rank{rank}-{quantize}-{routed}.amdgcn").write_text(
        assembly
    )
    if not routed:
        # The routed kernel also writes the SiTU input through in BF16; it
        # publishes with the same helper.
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
    for seed, rows in enumerate(
        (4, 2, 64, 6, 16, 17, 3, 8, 192, 32, 5, 128, 4, 64, 32)
    ):
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
    assert not state.can_latent_input_gather(hidden_for(193, 101), weight)
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


def _latent_gather_routed_worker(rank: int, port: int) -> None:
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
        _check_latent_gather_routed(rank, device, group)
    finally:
        dist.destroy_process_group()


def _check_latent_gather_routed(
    rank: int, device: torch.device, group: dist.ProcessGroup
) -> None:
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        K3MoeRouting,
        prepare_k3_communication,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.latent_gather import (
        iris_k3moe_latent_input_gather_routed,
    )
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sigmoid_bias_topk_sorted,
        kimi3_sorted_routing_scratch,
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
    generator = torch.Generator(device=device).manual_seed(20261115)
    weight = (
        torch.randn((_PACKED, _HIDDEN), generator=generator, device=device) / 40
    ).to(torch.bfloat16)
    weight[_ROUTER + _LATENT :] = (
        torch.randn((2 * _SHARED, _HIDDEN), device=device) / 40
    ).to(torch.bfloat16)
    down_weight = (
        torch.randn((_HIDDEN, _SHARED), generator=generator, device=device) / 30
    ).to(torch.bfloat16)
    # A coarse bias repeats values, so zero rows tie many experts' choices.
    bias = (
        torch.round(torch.randn((_ROUTER,), generator=generator, device=device) * 4)
        / 80
    )
    beta, linear_beta, block_size = 25.0, 25.0, 16
    reference_scratch = kimi3_sorted_routing_scratch(device)
    routing = K3MoeRouting(
        bias, 1.0, True, block_size, kimi3_sorted_routing_scratch(device)
    )

    def reference(hidden, quantize):
        router, routed, situ, *fp8 = state.latent_input_gather(
            hidden, weight, beta=beta, linear_beta=linear_beta, quantize=quantize
        )
        *routes, shared = kimi3_sigmoid_bias_topk_sorted(
            router,
            bias,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
            block_size=block_size,
            scratch=reference_scratch,
            shared_down=(situ, down_weight, None),
        )
        return router, routed, fp8[0] if fp8 else None, routes, shared

    def fused(hidden, quantize):
        assert state.can_latent_input_gather_routed(
            hidden, weight, routing, down_weight, None
        )
        router, routed, routes, shared, fp8 = state.latent_input_gather_routed(
            hidden,
            weight,
            beta=beta,
            linear_beta=linear_beta,
            routing=routing,
            down_weight=down_weight,
            quantize=quantize,
        )
        return router, routed, fp8, routes, shared

    def bits(tensor):
        if tensor.dtype == torch.float8_e4m3fn:
            return tensor.view(torch.uint8)
        return tensor.view(
            torch.int16 if tensor.dtype != torch.float32 else torch.int32
        )

    def same(a, b):
        return torch.equal(bits(a), bits(b))

    def valid_sort(routes):
        _, _, sorted_ids, sorted_weights, sorted_expert_ids, num_valid = routes
        padded = int(num_valid[0])
        ids = sorted_ids[:padded]
        real = (ids >> 24) < 16
        return (
            num_valid,
            ids,
            sorted_expert_ids[: padded // block_size],
            torch.where(real, sorted_weights[:padded], 0.0),
        )

    def check(got, want, label):
        router, routed, fp8, routes, shared = got
        want_router, want_routed, want_fp8, want_routes, want_shared = want
        assert same(router, want_router), label
        assert same(routed, want_routed), label
        if want_fp8 is not None:
            assert same(fp8[0], want_fp8[0]) and same(fp8[1], want_fp8[1]), label
        assert torch.equal(routes[1], want_routes[1]), label
        got_sort, want_sort = valid_sort(routes), valid_sort(want_routes)
        for a, b in zip(got_sort[:3], want_sort[:3], strict=True):
            assert torch.equal(a, b), label
        torch.testing.assert_close(routes[0], want_routes[0], rtol=2e-7, atol=0)
        torch.testing.assert_close(got_sort[3], want_sort[3], rtol=2e-7, atol=0)
        torch.testing.assert_close(
            shared.float(), want_shared.float(), rtol=1e-2, atol=1e-2
        )

    hidden_rows = {}

    def hidden_for(rows, seed):
        if seed not in hidden_rows:
            source = torch.Generator(device=device).manual_seed(seed)
            hidden = torch.randn((rows, _HIDDEN), generator=source, device=device)
            # Zero rows tie every expert's logit: the lower expert goes first.
            hidden[seed % rows] = 0.0
            hidden_rows[seed] = hidden.to(torch.bfloat16)
        return hidden_rows[seed]

    producer = state.all_reduce
    for seed, rows in enumerate((8, 2, 16, 6, 3, 12, 8, 5, 4, 15, 8)):
        hidden = hidden_for(rows, seed)
        quantize = seed % 3 != 1
        want = reference(hidden, quantize)
        check(fused(hidden, quantize), want, f"rows={rows} seed={seed}")
        # Interleave with the MoE all-reduce, which shares the mailbox tiles.
        reduce_rows = min(rows, 6)
        outputs = producer.acquire_outputs(
            ((reduce_rows, _LATENT), (reduce_rows, _HIDDEN))
        )
        for output in outputs:
            output.fill_(1.0)
        reduced = state.reduce_moe(outputs)
        assert torch.equal(reduced[0], torch.full_like(reduced[0], 8.0))
    assert not state.can_latent_input_gather_routed(
        hidden_for(17, 300), weight, routing, down_weight, None
    )
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])

    # Captured decode layers with one rank late, sharing the routing scratch
    # with the separate routing kernel.
    layers = 6
    shared_routing = K3MoeRouting(bias, 1.0, True, block_size, reference_scratch)
    expected = [
        reference(hidden_for(8 + layer % 2 * 4, 400 + layer), layer % 3 != 2)
        for layer in range(layers)
    ]

    def captured(layer):
        hidden = hidden_for(8 + layer % 2 * 4, 400 + layer)
        if layer % 2:
            return reference(hidden, layer % 3 != 2)
        router, routed, routes, shared, fp8 = state.latent_input_gather_routed(
            hidden,
            weight,
            beta=beta,
            linear_beta=linear_beta,
            routing=shared_routing,
            down_weight=down_weight,
            quantize=layer % 3 != 2,
        )
        return router, routed, fp8, routes, shared

    def flat(result):
        router, routed, fp8, routes, shared = result
        return [router, routed, *(fp8 or ()), *routes, shared]

    for layer in range(layers):
        captured(layer)
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    snapshots = None
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), assert_no_triton_compile(
        iris_k3moe_latent_input_gather_routed
    ):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            results = []
            for layer in range(layers):
                if rank == (layer * 3 + 1) % 8:
                    torch.cuda._sleep(100_000)
                results.append(flat(captured(layer)))
            snapshots = [[t.clone() for t in layer] for layer in results]
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for layer, (snapshot, want) in enumerate(zip(snapshots, expected, strict=True)):
            router, routed, *rest = snapshot
            fp8 = tuple(rest[:2]) if layer % 3 != 2 else None
            routes = rest[2:8] if fp8 else rest[:6]
            check(
                (router, routed, fp8, routes, rest[-1]), want, f"replay layer {layer}"
            )
    dist.barrier(group=group, device_ids=[rank])


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="the K3 latent-input gather runs across eight CDNA4 GPUs",
)
def test_iris_moe_latent_input_gather_routed() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_latent_gather_routed_worker, args=(port,), nprocs=8, join=True)
